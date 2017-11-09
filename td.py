from utils import v

def TDinf_targets(episodes):
    """Generate td_targets (TDinf).
    episodes = (episode, ..)
    episode = (state, action, reward, newstate)
    Events in episode must come in order. That is event[0].newstate == event[1].state.
    """
    discount = 0.95
    for episode in episodes:
        episode = list(episode)
        # Work backwards and calculate td_targets
        td_target = 0.0  # assuming episode is a full rollout
        for state, action, reward, _ in episode[::-1]:
            td_target = reward + discount*td_target
            yield ((*state, action), td_target)


def TD0_targets(episodes, q):
    discount = 0.95
    for episode in episodes:
        for state, action, reward, newstate in episode:
            td_target = reward + discount*v(q, newstate)
            yield ((*state, action), td_target)
