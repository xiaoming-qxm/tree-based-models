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
  // Caculate best information gain
  int BestInfoGain(const std::vector<int> data_idx, const std::vector<int> feat_idx,
                           const std::vector<int> data, const std::vector<int> labels,
                           const int num_classes);

  // Caculate best information gain ratio
  int BestInfoGainRatio(const std::vector<int> data_idx, const std::vector<int> feat_idx,
                           const std::vector<int> data, const std::vector<int> labels,
                           const int num_classes);

  ~Entropy() = default;

};

}

#endif // ENTROPY_H_
