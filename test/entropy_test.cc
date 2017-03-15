// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#include "../criterion/entropy.h"
#include <vector>
#include <iostream>

using namespace std;
namespace tbm = tree_based_model;

int main(void)
{
  tbm::Entropy ent;
  // test sample
  int num_classes = 2;
  // Example in Method of Statistical Learining, Li Hang
  const vector<int> data{0,0,0,0,
                         0,0,0,1,
                         0,1,0,1,
                         0,1,1,0,
                         0,0,0,0,
                         1,0,0,0,
                         1,0,0,1,
                         1,1,1,1,
                         1,0,1,2,
                         1,0,1,2,
                         2,0,1,2,
                         2,0,1,1,
                         2,1,0,1,
                         2,1,0,2,
                         2,0,0,0};

  const vector<int> labels{0,0,1,1,0,0,0,1,1,1,1,1,1,1,0};

  vector<int> feat_idx{0,1,2,3};
  vector<int> data_idx;
  for(int i = 0; i < 15; ++i)
    data_idx.push_back(i);

  int best_feature_idx = 0;
  best_feature_idx = ent.BestInfoGain(data_idx, feat_idx, data, labels, num_classes);
  cout << "Use info gain, best feature index: " << best_feature_idx << endl;
  cout << "Impurity value:  " << ent.get_impurity() << endl;

  best_feature_idx = ent.BestInfoGainRatio(data_idx, feat_idx, data, labels, num_classes);
  cout << "Use info gain ratio, best feature index: " << best_feature_idx << endl;
  cout << "Impurity value:  " << ent.get_impurity() << endl;
  return 0;
}
