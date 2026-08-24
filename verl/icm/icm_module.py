import torch
import torch.nn.functional as F


def whiten(values):  #SOO: token level ICM
    mean = torch.mean(values)
    var = torch.var(values, unbiased=False)
    return (values - mean) * torch.rsqrt(var + 1e-8)


class ForwardModel(torch.nn.Module):
    '''
    predict next_state_hat using state and action
    '''
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        # self.action_encoder = torch.nn.Linear(2048, 2048)
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, intermediate_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(intermediate_dim, intermediate_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(intermediate_dim, hidden_dim)
        )
    
    def forward(self, state, action):
        # action = self.action_encoder(action)
        x = torch.cat([state, action], dim=-1)
        x = self.hidden(x)
        return x

class ICM(torch.nn.Module):
    def __init__(self, hidden_dim, intermediate_dim) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim * 2, hidden_dim * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.forward_model = ForwardModel(hidden_dim, intermediate_dim)
    
    def forward(self, state, next_state, action):
        state = self.encoder(state)
        next_state = self.encoder(next_state)
        next_state_hat = self.forward_model(state, action)
        return next_state, next_state_hat
    
    def prepare_icm_input(self, ref_hidden_states, actor_step_embs):
        """
        ref_hidden_states: list of (num_steps, hidden_dim)  - s_1, s_2, ...
        actor_step_embs:   list of (num_steps, hidden_dim)  - a_1, a_2, ...
        """
        bs = len(ref_hidden_states)

        s_t_list, a_t_list, s_t1_list = [], [], []
        pair_map = []

        for i in range(bs):
            s = ref_hidden_states[i]  # (num_steps+1, hidden_dim)
            a = actor_step_embs[i]    # (num_steps, hidden_dim)
            num_steps = a.shape[0]

            for t in range(num_steps):
                s_t_list.append(s[t])
                a_t_list.append(a[t])
                s_t1_list.append(s[t + 1])
                pair_map.append((i, t))
        
        # 배치로 묶기
        s_t_batch = torch.stack(s_t_list, dim=0)
        a_t_batch = torch.stack(a_t_list, dim=0)
        s_t1_batch = torch.stack(s_t1_list, dim=0)

        return s_t_batch, a_t_batch, s_t1_batch, pair_map

    def icm_loss(self, next_state, next_state_hat):
        forward_model_loss = 0.5 * torch.sum((next_state_hat - next_state).norm(2, dim=-1).pow(2)) / (len(next_state) + 1e-8)

        return forward_model_loss