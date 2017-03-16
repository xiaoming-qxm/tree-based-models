// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#include "entropy.h"
#include <unordered_map>
#include <vector>
#include <cmath>
#include <iostream>

namespace tree_based_model {

int Entropy::FeatWithBestIG(const std::vector<int>& data_idx, std::vector<int>& feat_idx,
                            const std::vector<int>& data, const std::vector<int>& labels,
                            const int num_classes, const int num_feature) {
  std::unordered_map<int, int> class_map;
  // temp is a placeholder for vector<vector<int>> type
  std::vector<int> temp;
  temp.resize(num_classes);
  double empirical_entropy = 0.;

  // Count per class number
  for(unsigned i = 0; i < labels.size(); ++i) {
    if(class_map.find(labels[i]) == class_map.end())
      class_map[labels[i]] = 1;
    else
      class_map[labels[i]] += 1;
  }

  // number of data
  int num_data = labels.size();
  double p = 0.;

  // Caculate empirical entropy
  for(int i = 0; i < num_classes; ++i) {
    p = static_cast<double>(class_map[i]) / num_data;
    empirical_entropy += - p * std::log2(p);
  }
  // so far so good

  int best_feature = 0;
  int best_feat_idx = 0;
  int num_row = data_idx.size();
  int num_col = feat_idx.size();
  double best_info_gain = -1000.;

  for(int i = 0; i < num_col; ++i) {
    std::unordered_map<int, int> num_data_grp;
    std::unordered_map<int, int> feat_map;
    std::vector<std::vector<int>> cls_num_per_grp;

    int counter = 0, idx = 0;
    for(int j = 0; j < num_row; ++j) {
      idx = data_idx[j] * num_feature + feat_idx[i];
      if(feat_map.find(data[idx]) == feat_map.end()) {
        // std::cout << "data " << j << ""
        feat_map[data[idx]] = counter;
        // temp is a placeholder in here
        cls_num_per_grp.push_back(temp);
        cls_num_per_grp[counter][labels[j]] = 1;
        num_data_grp[counter] = 1;
        counter++;
      }else {
        cls_num_per_grp[feat_map[data[idx]]][labels[j]] += 1;
        num_data_grp[feat_map[data[idx]]] += 1;
      }
    }

    double info_gain = empirical_entropy;
    for(unsigned j = 0; j < cls_num_per_grp.size(); ++j) {
      double cond_ent = 0.;
      for(int k = 0; k < num_classes; ++k) {
        p = static_cast<double>(cls_num_per_grp[j][k]) / num_data_grp[j];
        if(p != 0.)
          cond_ent += - p * std::log2(p);
      }
      info_gain -= num_data_grp[j] * cond_ent / num_data;
    }

    // std::cout << "info ratio " << info_gain << std::endl;

    if(info_gain > best_info_gain) {
      best_feature = feat_idx[i];
      best_feat_idx = i;
      best_info_gain = info_gain;
    }
  }

  // Remove best feature from feature set
  feat_idx.erase(feat_idx.begin() + best_feat_idx);
  impurity_value = best_info_gain;

  return best_feature;
}

int Entropy::FeatWithBestIGR(const std::vector<int>& data_idx, std::vector<int>& feat_idx,
                              const std::vector<int>& data, const std::vector<int>& labels,
                              const int num_classes, const int num_feature) {
  std::unordered_map<int, int> class_map;
  // temp is a placeholder for vector<vector<int>> type
  std::vector<int> temp;
  temp.resize(num_classes);
  double empirical_entropy = 0.;
  std::vector<double> info_gain_arr;
  // the entropy array of the features
  std::vector<double> feat_ent_arr;

  // Count per class number
  for(unsigned i = 0; i < labels.size(); ++i) {
    if(class_map.find(labels[i]) == class_map.end())
      class_map[labels[i]] = 1;
    else
      class_map[labels[i]] += 1;
  }

  // number of data
  int num_data = labels.size();
  double p = 0.;

  // Caculate empirical entropy
  for(int i = 0; i < num_classes; ++i) {
    p = static_cast<double>(class_map[i]) / num_data;
    empirical_entropy += - p * std::log2(p);
  }
  // so far so good

  int best_feature = 0;
  int best_feat_idx = 0;
  int num_row = data_idx.size();
  int num_col = feat_idx.size();
  double best_info_gain_ratio = -1000.;
  double mean_info_gain = 0.;

  for(int i = 0; i < num_col; ++i) {
    std::unordered_map<int, int> num_data_grp;
    std::unordered_map<int, int> feat_map;
    std::vector<std::vector<int>> cls_num_per_grp;

    int counter = 0, idx = 0;
    for(int j = 0; j < num_row; ++j) {
      idx = data_idx[j] * num_feature + feat_idx[i];
      if(feat_map.find(data[idx]) == feat_map.end()) {
        // std::cout << "data " << j << ""
        feat_map[data[idx]] = counter;
        // temp is a placeholder in here
        cls_num_per_grp.push_back(temp);
        cls_num_per_grp[counter][labels[j]] = 1;
        num_data_grp[counter] = 1;
        counter++;
      }else {
        cls_num_per_grp[feat_map[data[idx]]][labels[j]] += 1;
        num_data_grp[feat_map[data[idx]]] += 1;
      }
    }

    double info_gain = empirical_entropy;
    double feat_entropy = 0.;

    for(unsigned j = 0; j < cls_num_per_grp.size(); ++j) {
      double cond_ent = 0.;
      for(int k = 0; k < num_classes; ++k) {
        p = static_cast<double>(cls_num_per_grp[j][k]) / num_data_grp[j];
        if(p != 0.)
          cond_ent += - p * std::log2(p);
      }
      info_gain -= num_data_grp[j] * cond_ent / num_data;

      // Entropy of the feature
      p = static_cast<double>(num_data_grp[j]) / num_data;
      if(p != 0.)
        feat_entropy += - p * std::log2(p);
    }

    info_gain_arr.push_back(info_gain);
    feat_ent_arr.push_back(feat_entropy);
    // info gain mean value of all the feature
    mean_info_gain += info_gain / num_col;

    // std::cout << "info gain     " << info_gain << std::endl;
    // std::cout << "feat entropy: " << feat_entropy << std::endl;
  }

  // std::cout << "mean info gain  " << mean_info_gain << std::endl;

  for(int i = 0; i < num_col; ++i) {
    if(feat_ent_arr[i] == 0) continue;

    if(info_gain_arr[i] > mean_info_gain) {
      if((info_gain_arr[i] / feat_ent_arr[i]) > best_info_gain_ratio) {
        best_info_gain_ratio = info_gain_arr[i] / feat_ent_arr[i];
        best_feature = feat_idx[i];
        best_feat_idx = i;
      }
    }
  }

  // Remove best feature from feature set
  feat_idx.erase(feat_idx.begin() + best_feat_idx);
  impurity_value = best_info_gain_ratio;

  return best_feature;
}

}
