// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#ifndef CRITERION_H_
#define CRITERION_H_

namespace tree_based_model {

class Criterion {
public:
  Criterion() = default;
  // The impurity of the node
  double get_impurity() const { return impurity_value; }

  virtual ~Criterion() = default;

protected:
  double impurity_value;

};

}

#endif // CRITERION_H_
