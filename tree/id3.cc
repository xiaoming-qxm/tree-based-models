// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#include "id3.h"
#include "../criterion/entropy.h"
#include <vector>
#include <stdexcept>
#include <queue>
#include <unordered_map>
#include <iostream>

namespace tree_based_model {

void ID3::Fit(const std::vector<int>& data, const std::vector<int>& labels) {
  BuildTree(data, labels);
}

bool ID3::IsSameClass(const std::vector<int>& labels, const std::vector<int>& data_idx) {
  int u = labels[data_idx[0]];
  for(unsigned i = 0; i < data_idx.size(); ++i)
    if(u != labels[data_idx[i]])
      return false;
  return true;
}

int ID3::FindMaxClass(const std::vector<int>& labels, const std::vector<int>& data_idx) {
  std::vector<int> vec(num_classes, 0);

  for(unsigned i = 0; i < data_idx.size(); ++i)
    vec[labels[data_idx[i]]] += 1;

  // Debug
  std::cout << "class distribution: ";
  for(int i = 0; i < num_classes; ++i) {
    std::cout << vec[i] << " ";
  }

  int max_class = 0;
  int max_num = 0;
  for(int i = 0; i < num_classes; ++i) {
    if(vec[i] > max_num) {
      max_num = vec[i];
      max_class = i;
    }
  }
  return max_class;
}

void ID3::BuildTree(const std::vector<int>& data, const std::vector<int>& labels) {
  if(data.size() % num_feature != 0 || data.size() == 0)
    throw std::runtime_error("Error: Invalid feature number or data size.");
  if(data.size() / num_feature != labels.size())
    throw std::runtime_error("Error: data and labels size must be same.");

  int best_feat;
  int global_step = 0;
  std::vector<int> all_data_idx;
  std::vector<int> data_idx;
  std::vector<int> feat_idx;
  std::queue<std::vector<int>> data_queue;

  // initially
  for(unsigned i = 0; i < data.size() / num_feature; ++i)
    all_data_idx.push_back(i);
  for(int i = 0; i < num_feature; ++i)
    feat_idx.push_back(i);

  Entropy ent;

  int iter = 0;
  while(!data_queue.empty() || global_step == 0) {
    if(global_step == 0)
      data_idx = all_data_idx;
    else {
      data_idx = data_queue.front();
      data_queue.pop();
    }

    std::cout << std::endl <<  "..........     iter "
                           << iter << "    .........."
                           << std::endl;
    ++iter;

    Node node;

    // the feature set is empty
    if(feat_idx.empty()) {
      node.is_leaf = true;
      node.class_id = FindMaxClass(labels, data_idx);
      tree.push_back(node);
      continue;
    }

    // all dataset belong to the same class
    if(IsSameClass(labels, data_idx)) {
      node.is_leaf = true;
      node.class_id = FindMaxClass(labels, data_idx);
      tree.push_back(node);
      continue;
    }

    best_feat = ent.FeatWithBestIG(data_idx, feat_idx,
                                    data, labels,
                                    num_classes, num_feature);
    // Debug
    std::cout << "best feature: " << best_feat << std::endl;
    std::cout << "Remaining feature: ";
    for(unsigned i = 0; i < feat_idx.size(); ++i)
      std::cout << feat_idx[i] << " ";
    std::cout << std::endl;
    std::cout << "Impurity: " << ent.get_impurity() << std::endl;

    if(ent.get_impurity() < epsilon) {
      node.is_leaf = true;
      node.class_id = FindMaxClass(labels, data_idx);
      tree.push_back(node);
      continue;
    }

    std::unordered_map<int, int> feat_sub_data_map;
    std::unordered_map<int, int> child_node_map;
    // temp is a place holder
    std::vector<int> temp;
    std::vector<std::vector<int>> sub_data_set;
    int cnt = 0;
    int data_val;
    for(unsigned i = 0; i < data_idx.size(); ++i) {
      data_val = data[data_idx[i] * num_feature + best_feat];

      if(feat_sub_data_map.find(data_val) == feat_sub_data_map.end()) {
        ++global_step;
        child_node_map[data_val] = global_step;
        feat_sub_data_map[data_val] = cnt;
        sub_data_set.push_back(temp);
        sub_data_set[cnt].push_back(data_idx[i]);
        ++cnt;
      } else {
        sub_data_set[feat_sub_data_map[data_val]].push_back(data_idx[i]);
      }
    }

    for(unsigned i = 0; i < sub_data_set.size(); ++i) {
      data_queue.push(sub_data_set[i]);
    }

    sub_data_set.clear();

    for(std::unordered_map<int, int>::iterator iter = child_node_map.begin(); iter != child_node_map.end(); ++iter) {
      std::cout << "key: " << iter->first << " value: " << iter->second << std::endl;
    }

    node.child_node_map = child_node_map;
    node.feature_id = best_feat;
    node.is_leaf = false;
    node.class_id = FindMaxClass(labels, data_idx);
    node.impurity = ent.get_impurity();

    tree.push_back(node);
  }

}

}
