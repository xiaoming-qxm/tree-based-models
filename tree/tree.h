// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#ifndef TREE_MODEL_H_
#define TREE_MODEL_H_

#include<vector>
#include<string>

namespace tree_based_model {

class TreeModel {
public:
  virtual ~TreeModel() = default;
  // train model
  virtual void Fit(const std::vector<int>& data, const std::vector<int>& labels) = 0;
  // load model from json
  virtual void LoadModel(const std::string file_name) = 0;
  // save model to json
  virtual void SaveModel(const std::string file_name) = 0;
  // predict
  virtual void Predict(const std::vector<int>& data) = 0;
  // Evaluate
  virtual void Evaluate(const std::vector<int>& data, const std::vector<int>& labels) = 0;

};

}

#endif // TREE_MODEL_H_
