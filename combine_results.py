import json

# Modify the path of the result
audio_name = "audio-wavLM-epoch3-nms0"
video_name = "video-CLIP-epoch3-nms0"

# Get prediction_track2.txt
def read_prediction(file_path):
    pred = {}
    with open(file_path, 'r') as f:
        for line in f:
            vid, score = line.strip().split(',')
            pred[vid] = float(score)
    return pred
pred1 = read_prediction(f'results/{audio_name}/prediction.txt')
pred2 = read_prediction(f'results/{video_name}/prediction.txt')
merged_pred = {}
for vid in set(pred1.keys()).union(set(pred2.keys())):
    score1 = pred1.get(vid, 0.0)
    score2 = pred2.get(vid, 0.0)
    merged_pred[vid] = max(score1, score2)
with open('prediction/prediction_track2.txt', 'w') as f:
    for vid, score in merged_pred.items():
        f.write(f"{vid},{score:.4f}\n")

# Get prediction.json
with open(f'results/{audio_name}/prediction.json', 'r') as f:
    dict1 = json.load(f)
with open(f'results/{video_name}/prediction.json', 'r') as f:
    dict2 = json.load(f)
merged_dict = {}
for vid in set(dict1.keys()).union(set(dict2.keys())):
    segments = []
    if vid in dict1:
        segments.extend(dict1[vid])
    if vid in dict2:
        segments.extend(dict2[vid])
    merged_dict[vid] = segments
with open('prediction/prediction.json', 'w') as f:
    json.dump(merged_dict, f, indent=4)


