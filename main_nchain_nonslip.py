import gym  # OpenAI gym

import glue


def main():
    env = gym.make('NChain-v0')
    env.unwrapped.slip = 0  # nonslip env
    glue.run(env)


if __name__ == '__main__':
    main()
