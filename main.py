import argparse
from core.example1 import Example1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type_app", type=str, help='Typ App', required=True)
    ap.add_argument("-c", "--category", type=str, help='Category', required=True)
    ap.add_argument("-vs", "--video_source", type=str, help='Video Source', required=False)

    args = vars(ap.parse_args())
    type = args["type_app"]  # type
    category = args["category"]
    video_source = args["video_source"]  # video source

    main = Example1()
    main.still_running = True
    main.label = category
    main.run_main(type, video_source)



