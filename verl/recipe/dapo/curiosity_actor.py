import torch
import ray
from transformers import get_scheduler


@ray.remote(num_gpus=1)
class CuriosityActor:
    def __init__(self, prm_model_path, hidden_size, intermediate_size, icm_config, total_training_steps):
        from recipe.dapo.prm_scorer import PRMScorer
        from icm.icm_module import ICM

        # PRM
        self.prm_scorer = PRMScorer(prm_model_path)
        print(f"[PRM-DEBUG] PRM initialized on {self.prm_scorer.device}", flush=True)

        # ICM
        self.icm = ICM(hidden_size, intermediate_size).to(torch.bfloat16).cuda()
        self.icm_optimizer = torch.optim.Adam(
            self.icm.parameters(),
            lr=icm_config.lr,
            betas=(0.9, 0.95),
        )
        self.icm_lr_scheduler = get_scheduler(
            name=icm_config.lr_scheduler_type,
            optimizer=self.icm_optimizer,
            num_warmup_steps=icm_config.warmup_steps,
            num_training_steps=total_training_steps,
        )
        print(f"[ICM-DEBUG] ICM initialized on {next(self.icm.parameters()).device}", flush=True)

    def score_prm_batch(self, problem_str_list, steps_list):
        all_labels = []
        for problem_str, steps in zip(problem_str_list, steps_list):
            if steps:
                labels = self.prm_scorer.score_steps(problem_str, steps)
            else:
                labels = []
            all_labels.append(labels)
        return all_labels

    def compute_icm(self, ref_hidden_states, actor_step_embs):
        """
        ref_hidden_states: list of (num_steps, hidden_dim) tensors
        actor_step_embs: list of (num_steps, hidden_dim) tensors
        returns: intrinsic_reward (num_pairs,), pair_map
        """
        s_t_batch, a_t_batch, s_t1_batch, pair_map = self.icm.prepare_icm_input(
            ref_hidden_states, actor_step_embs
        )
        s_t_batch = s_t_batch.to(torch.bfloat16).cuda()
        a_t_batch = a_t_batch.to(torch.bfloat16).cuda()
        s_t1_batch = s_t1_batch.to(torch.bfloat16).cuda()

        next_state, next_state_hat = self.icm(s_t_batch, s_t1_batch, a_t_batch)

        icm_loss = self.icm.icm_loss(next_state, next_state_hat)

        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()
        self.icm_lr_scheduler.step()

        icm_loss_val = icm_loss.item()
        del s_t_batch, a_t_batch, s_t1_batch, icm_loss

        intrinsic_reward = 0.5 * (next_state - next_state_hat).norm(2, dim=-1)
        mean = intrinsic_reward.mean()
        var = intrinsic_reward.var(unbiased=False)
        intrinsic_reward = (intrinsic_reward - mean) * torch.rsqrt(var + 1e-8)
        intrinsic_reward = intrinsic_reward.detach().cpu()

        del next_state, next_state_hat
        torch.cuda.empty_cache()

        return intrinsic_reward, pair_map, icm_loss_val