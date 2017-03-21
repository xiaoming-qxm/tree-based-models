// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#ifndef ID_3_H_
#define ID_3_H_

#include "tree.h"

#include <vector>
#include <string>
#include <unordered_map>

namespace tree_based_model {

struct Node {

  bool is_leaf = false;
  // -1 for no feature until now
  int feature_id = -1;
  // -1 for no belonging class until now
  int class_id = -1;
  // -1 for no parents until now
  int parents_node_id = -1;
  // feature impurity
  double impurity;
  // Map from children index to corresponding node index
  std::unordered_map<int, int> child_node_map;

};

class ID3 : public TreeModel {
public:
  ID3(const int n_cls, const int n_feat) :
      num_classes(n_cls), num_feature(n_feat) { epsilon = 0.1; }
  ID3(const int n_cls, const int n_feat, const double eps) :
      num_classes(n_cls), num_feature(n_feat), epsilon(eps) { }

  // train model
  void Fit(const std::vector<int>& data, const std::vector<int>& labels);
  // predict
  void Predict(const std::vector<int>& data) {}
  // Evaluate
  void Evaluate(const std::vector<int>& data, const std::vector<int>& labels) {}
  // load model from json
  void LoadModel(const std::string file_name);
  // save model to json
  void SaveModel(const std::string file_name);

private:
  // whether is all same class or not
  bool IsSameClass(const std::vector<int>& labels, const std::vector<int>& data_idx);
  // find class which has largest number in the dataset
  int FindMaxClass(const std::vector<int>& labels, const std::vector<int>& data_idx);
  // build tree
  void BuildTree(const std::vector<int>& data, const std::vector<int>& labels);
  // pruning tree
  void Pruning(const std::vector<int>& data, const std::vector<int>& labels);


private:
  int num_classes;
  int num_feature;
  // minimum information gain value
  double epsilon;

  std::vector<Node> tree;

};

}


#endif // ID_3_H_
