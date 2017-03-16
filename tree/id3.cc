// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#include "id3.h"
#include "../criterion/entropy.h"
#include <vector>
#include <stdexcept>
#include <queue>
#include <unordered_map>

namespace tree_based_model {

void ID3::Fit(const std::vector<int>& data, const std::vector<int>& labels) {
  BuildTree(data, labels);
}

void ID3::BuildTree(const std::vector<int>& data, const std::vector<int>& labels) {
  if(data.size() % num_feature != 0 || data.size() == 0)
    throw std::runtime_error("Error: Invalid feature number or data size.");
  if(data.size() != labels.size())
    throw std::runtime_error("Error: data and labels size must be same.");

  int best_feat;
  int global_step = 0;
  std::vector<int> all_data_idx;
  std::vector<int> feat_idx;
  std::queue<std::vector<int>> data_queue;

  data_queue.push(all_data_idx);

  // initially
  for(unsigned i = 0; i < data.size() / num_feature; ++i)
    all_data_idx.push_back(i);
  for(int i = 0; i < num_feature; ++i)
    feat_idx.push_back(i);

  int start_point = 0;
  Entropy ent;

  while(!data_queue.empty()) {
    std::vector<int> &data_idx = data_queue.front();
    Node node(data_idx, start_point);

    // all dataset belong to the same class
    // TODO
    if(false) {
      node.is_leaf = true;
      node.class_id = 0;
      continue;
    }
    // the feature set is empty
    if(feat_idx.size() == 0) {
      node.is_leaf = true;
      // TODO
      node.class_id = 0;
      continue;
    }

    best_feat = ent.FeatWithBestIG(data_idx, feat_idx,
                                    data, labels,
                                    num_classes, num_feature);
    if(ent.get_impurity() < epsilon) {
      node.is_leaf = true;
      // TODO
      node.class_id = 0;
      continue;
    }

    std::unordered_map<int, int> feat_sub_data_map;
    // temp is a place holder
    std::vector<int> temp;
    std::vector<std::vector<int>> sub_data_set;
    int cnt = 0;
    for(unsigned i = 0; i < data_idx.size(); ++i) {
      int val = data[i * num_classes + best_feat];
      if(feat_sub_data_map.find(val) == feat_sub_data_map.end()) {
        feat_sub_data_map[val] = cnt;
        sub_data_set.push_back(temp);
        sub_data_set[cnt].push_back(labels[i * num_classes + best_feat]);
        ++cnt;
      } else {
        sub_data_set[feat_sub_data_map[val]].push_back(labels[i * num_classes + best_feat]);
      }
    }

    // TODO
    for(unsigned i = 0; i < sub_data_set.size(); ++i) {
      data_queue.push(sub_data_set[i]);
    }

    tree.push_back(node);
    data_queue.pop();
  }

}

}
