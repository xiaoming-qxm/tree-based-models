// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#ifndef ENTROPY_H_
#define ENTROPY_H_

#include <vector>
#include "classification_criterion.h"

namespace tree_based_model {

// Cross Entropy impurity criterion
class Entropy : public ClassificationCriterion {
public:
  Entropy() = default;
  // Get split feature with best information gain, only on discrete features
  int FeatWithBestIG(const std::vector<int>& data_idx, std::vector<int>& feat_idx,
                              const std::vector<int>& data, const std::vector<int>& labels,
                              const int num_classes, const int num_feature);

  // Get split feature best information gain ratio only on discrete features
  int FeatWithBestIGR(const std::vector<int>& data_idx, std::vector<int>& feat_idx,
                                    const std::vector<int>& data, const std::vector<int>& labels,
                                    const int num_classes, const int num_feature);

  ~Entropy() = default;

};

}

#endif // ENTROPY_H_
