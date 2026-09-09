"""Offline capture comparison without importing research package infrastructure."""
if __package__:
    from .check_yank_deployed_replay import load_tool
else:
    from check_yank_deployed_replay import load_tool


def main():return load_tool('compare').main()

if __name__=='__main__':main()
