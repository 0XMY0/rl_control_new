#include <iostream>

#include "utils/history_buffer.hpp"

int main() {
  ovinf::HistoryBuffer<float> buffer(2, 4);
  Eigen::VectorXf obs(2);

  auto buffer_test = [&]() {
    auto obs_buffer = buffer.GetObsHistory();
    std::cout << obs_buffer.transpose() << std::endl;
    auto obs_short_buffer = buffer.GetObsHistory(3);
    std::cout << obs_short_buffer.transpose() << std::endl;
    std::cout << buffer.GetObservation(0).transpose() << std::endl;

    std::cout << std::endl;
  };

  obs << 1., 0.5;
  buffer.AddObservation(obs);
  buffer_test();

  obs << 2., 0.5;
  buffer.AddObservation(obs);
  buffer_test();

  obs << 3., 0.5;
  buffer.AddObservation(obs);
  buffer_test();

  obs << 4., 0.5;
  buffer.AddObservation(obs);
  buffer_test();

  obs << 5., 0.5;
  buffer.AddObservation(obs);
  buffer_test();

  obs << 6., 0.5;
  buffer.AddObservation(obs);
  buffer_test();

  return 0;
}
