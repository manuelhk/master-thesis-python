import pyrosbag as prb
import ros



with prb.BagPlayer("data/test.bag") as example:
    example.play()
    print(example)

