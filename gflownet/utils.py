import torch

def trajectory_balance_loss(total_flow, rewards, fwd_probs, back_probs):
    """
    Computes the mean trajectory balance loss for a collection of samples. For
    more information, see Bengio et al. (2022): https://arxiv.org/abs/2201.13259
    
    Args:
        total_flow: The estimated total flow used by the GFlowNet when drawing
        the collection of samples for which the loss should be computed
        
        rewards: The rewards associated with the final state of each of the
        samples
        
        fwd_probs: The forward probabilities associated with the trajectory of
        each sample (i.e. the probabilities of the actions actually taken in
        each trajectory)
        
        back_probs: The backward probabilities associated with each trajectory
    """
    
    loss = torch.square(torch.log(total_flow * torch.prod(fwd_probs, dim=1)) - torch.log((rewards * torch.prod(back_probs, dim=1))))
    return loss.mean()

    lhs = total_flow * torch.prod(fwd_probs, dim=1)
    rhs = rewards * torch.prod(back_probs, dim=1)
    loss = torch.log(lhs / rhs)**2
    return loss.mean()
