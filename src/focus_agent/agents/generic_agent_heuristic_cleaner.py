from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs, GenericAgent

from .utils import remove_no_bid_lines


class GenericAgentHeuristicCleanerArgs(GenericAgentArgs):
    def make_agent(self) -> GenericAgent:
        return GenericAgentHeuristicCleaner(self.chat_model_args, self.flags, self.max_retry)


class GenericAgentHeuristicCleaner(GenericAgent):
    def __init__(self, chat_model_args, flags, max_retry=4):
        super().__init__(chat_model_args, flags, max_retry)

    def obs_preprocessor(self, obs):
        obs = super().obs_preprocessor(obs)
        obs["axtree_txt"] = remove_no_bid_lines(obs["axtree_txt"])
        return obs
