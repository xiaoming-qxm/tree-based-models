// Copyright (c) 2017 The Authors. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file, See the AUTHORS file for names of contributors.

#ifndef CRITERION_H_
#define CRITERION_H_

using namespace tree_based_model {

class Criterion {
public:
  Criterion();
  // The impurity of the node
  virtual double NodeImpurity();
  // The impurity of the children
  virtual void ChildrenImpurity(double& impurity_left, double& impurity_right);
  // Node value
  virtual void NodeValue();
  // Inprovement in impurity after a split
  virtual double InpurityImprovement();

  ~Criterion();

private:
  double impurity_value;

};

}

#endif // CRITERION_H_
