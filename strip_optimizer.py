import torch
import os

weights = "/root/workspace/3-12-data/weights/best/pki_0526_5311.pth"
new_weights = "/root/workspace/3-12-data/weights/best/pki_0526_5311_small.pth"
x = torch.load(weights, map_location = torch.device('cpu'))

if x.get('ema'):
    x['model'] = x['ema']  # replace model with ema
for k in 'optimizer', 'best_fitness', 'ema', 'updates':  # keys
    x[k] = None
x['epoch'] = -1
for y in x:
    print(y)
# x['model'].half()  # to FP16
# for p in x['meta'].parameters():
#     p.requires_grad = False
torch.save(x, new_weights)
mb = os.path.getsize(new_weights) / 1E9  # filesize
print(mb)
