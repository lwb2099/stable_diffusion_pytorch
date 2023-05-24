def add_checkpoint_args(parser):
    checkpoint_group = parser.add_argument_group("checkpoint")
    checkpoint_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="dir to load checkpoints from",
    )
    checkpoint_group.add_argument(
        "--output_dir",
        type=str,
        default="dir to save and load checkpoints",
    )
