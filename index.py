from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--env", type=str, default="prod", help="Which environment to run? (dev or prod)")
parser.add_argument("--algo", type=str, default="mobilenet", help="Which algo? (mobilenet or yolo)")
parser.add_argument("--type", type=str, default="video", help="Which input? (image or video)")
args = parser.parse_args()

if args.env == "dev" and args.algo == "mobilenet" and args.type == "video":
    from mobilenet_video import start
elif args.env == "dev" and args.algo == "yolo":
    from yolo import start
elif args.env == "dev" and args.algo == "mobilenet" and args.type == "image":
    from mobilenet_image import start
elif args.env == "prod":
    from raspberry_video import start

start()


