from huggingface_hub import snapshot_download
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--token", type=str, default=None)

    return parser.parse_args()

def main():
    args = parse_arguments()
    snapshot_download(
        repo_id=args.model_dir,
        local_dir=args.output_dir,
        token=args.token
    )

if __name__ == "__main__":
    main()
