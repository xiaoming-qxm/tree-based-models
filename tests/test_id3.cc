// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#include "../base/model_factory.h"

#include <vector>
#include <iostream>

using namespace std;
namespace tbm = tree_based_model;

int main() {
  // test sample
  int num_classes = 2;
  int num_feature = 4;
  // Example in Method of Statistical Learining, by Li Hang
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

  tbm::ModelFactory mf;
  tbm::TreeModel* clf = mf.CreateModel("id3", num_classes, num_feature);

  clf->Fit(data, labels);
  clf->SaveModel("./model.json");
  clf->LoadModel("./model.json");

  return 0;
}
