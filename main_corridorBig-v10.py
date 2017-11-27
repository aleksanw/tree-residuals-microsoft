import corridor
import glue
import gym


def main():
    env = gym.make('CorridorBig-v10')
    glue.run(env)


if __name__ == '__main__':
    main()
