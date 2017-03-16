// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#ifndef ID_3_H_
#define ID_3_H_

#include <vector>
#include <unordered_map>

namespace tree_based_model {

class Node final {
public:
  Node(std::vector<int> child_idx, int start_point) {
    // -1 for no feature until now
    feature_id = -1;
    // -1 for no belonging class until now
    class_id = -1;
    // -1 for no parents until now
    parents_node_id = -1;

    for(unsigned i = 0; i < child_idx.size(), ++i)
      child_node_map[child_idx[i]] = start_point + i;
  }

  int feature_id;
  int class_id;
  int parents_node_id;
  // feature impurity
  double impurity;
  // Map from children index to corresponding node index
  std::unordered_map<int, int> child_node_map;

};

class ID3 final : public ClassificationTree {
public:
  ID3(const std::vector<int>& data, const int & labels,
      const int num_classes, const int num_feature);
  // train model
  void fit();
  // predict
  void predict();

private:
  const int num_classes;
  const int num_data;
  const int num_feature;

  const std::vector<int>& data;
  const int &labels;
  std::vector<Node> tree;

};

}


#endif // ID_3_H_
