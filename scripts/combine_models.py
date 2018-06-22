import torch, os, json, tarfile, argparse, uuid, shutil

def main(ques_dir, span_dir, outfile):
    ques_weights = torch.load(os.path.join(ques_dir, "best.th"))
    ques_config = json.loads(open(os.path.join(ques_dir, "config.json"), 'r').read())
    ques_model = ques_config['model']

    weights = {}
    for name, w in ques_weights.items():
        if name.startswith('span_hidden') or name.startswith('frame_'):
            continue

        if name.startswith('span_encoder'):
            name = name.replace('span_encoder', 'stacked_encoder')


        weights['question_predictor.' + name] = w

    span_weights = torch.load(os.path.join(span_dir, "best.th"))
    span_config = json.loads(open(os.path.join(span_dir, "config.json"), 'r').read())

    for name, w in span_weights.items():
        weights['span_detector.' + name] = w

    config = span_config
    config['model'] = {'type':'qasrl_parser', 'span_detector': span_config['model'], 'question_predictor' : ques_model}

    tmpdir = os.path.join("tmp", str(uuid.uuid4()))
    os.makedirs(tmpdir)

    weightfile = os.path.join(tmpdir, "weights.th")
    configfile = os.path.join(tmpdir, "config.json")
    vocabdir = os.path.join(tmpdir, "vocabulary")

    torch.save(weights, weightfile)
    with open(configfile, "w") as f:
        f.write(json.dumps(config))
    shutil.copytree(os.path.join(ques_dir, "vocabulary"), vocabdir)

    with tarfile.open(outfile, "w:gz") as tar:
        tar.add(weightfile, arcname="weights.th")
        tar.add(configfile, arcname="config.json")
        tar.add(vocabdir, arcname="vocabulary")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write CONLL format SRL predictions"
                                                 " to file from a pretrained model.")
    parser.add_argument('--ques', type=str)
    parser.add_argument('--span', type=str)
    parser.add_argument('--out', type=str)

    args = parser.parse_args()
    main(args.ques, args.span, args.out)
 
