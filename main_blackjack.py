import glue
import gym  # OpenAI gym


def main():
    env = gym.make('Blackjack-v0')
    glue.run(env)


if __name__ == '__main__':
    main()
