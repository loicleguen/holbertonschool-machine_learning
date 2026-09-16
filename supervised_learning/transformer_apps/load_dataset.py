#!/usr/bin/env python3
from setup import load_pt2en

pt2en_train = load_pt2en('train')
for pt, en in pt2en_train.take(1):
  print(pt.numpy().decode('utf-8'))
  print(en.numpy().decode('utf-8'))