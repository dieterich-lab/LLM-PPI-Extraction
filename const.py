from parser import args

if "ppi" in args.target:
    LOOKUP = "ppi"
elif "tf" in args.target:
    LOOKUP = "tf"
elif "lr" in args.target:
    LOOKUP = "lr"
else:
    LOOKUP = args.target

if args.target in ["ppi", "tf", "both"]:
    PAPERS = "ppi"
else:
    PAPERS = args.target
