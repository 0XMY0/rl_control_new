/**
 * @file history_buffer.
 * @brief History buffer for storing observations.
 * @author Dknt
 * @date 2025-4-15
 */

#ifndef HISTORY_BUFFER_HPP
#define HISTORY_BUFFER_HPP

#include <Eigen/Core>
#include <memory>

namespace ovinf {

/**
 * @brief Vector observation buffer. Does NOT support multi-threading!
 *        This is sort of ring buffer but not exactly. Make sure you understand
 *        what happens in this class. Use at your risk )
 *
 * @tparam T Data type. Scalar type only.
 */
template <typename T>
class HistoryBuffer {
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  template <typename U>
  using MapT = Eigen::Map<U>;

 public:
  using Ptr = std::shared_ptr<HistoryBuffer<T>>;

  /**
   * @brief Observation history buffer constructor.
   *        The buffer is initialized to 0.
   *
   * @param[in] single_obs_size Observation size
   * @param[in] buffer_size Buffer size
   */
  HistoryBuffer(size_t single_obs_size, size_t buffer_size) {
    if (single_obs_size <= 0 || buffer_size <= 0) {
      throw std::invalid_argument(
          "Buffer size must be greater than 0. HistoryBuffer(size_t, size_t)");
    }

    data_ = new T[single_obs_size * (buffer_size * 2 - 1)];
    single_obs_size_ = single_obs_size;
    buffer_size_ = buffer_size;

    current_index_ = -1;
    std::fill(data_, data_ + single_obs_size * (buffer_size * 2 - 1), 0);
  }

  ~HistoryBuffer() { delete[] data_; }

  /**
   * @brief Add a new observation to the buffer
   *
   * @param[in] obs Observation
   */
  void AddObservation(const VectorT& obs) {
    if (++current_index_ > buffer_size_ - 1) {
      current_index_ = 0;
    }
    MapT<VectorT>(data_ + current_index_ * single_obs_size_, single_obs_size_) =
        obs;
    if (current_index_ != buffer_size_ - 1)
      MapT<VectorT>(data_ + (current_index_ + buffer_size_) * single_obs_size_,
                    single_obs_size_) = obs;
  }

  /**
   * @brief Get the observation at the given index
   *
   * @param[in] index index, 0 is the newest
   * @return Obs
   */
  VectorT GetObservation(size_t index) {
    if (index >= buffer_size_) {
      throw std::out_of_range(
          "Index out of range. HistoryBuffer.GetObservation(size_t)");
    }
    VectorT obs(single_obs_size_);
    obs = MapT<VectorT>(
        data_ + (current_index_ - index) % buffer_size_ * single_obs_size_,
        single_obs_size_);
    return obs;
  }

  /**
   * @brief Get short history
   *
   * @param[in] size History size
   * @return Obs history
   */
  VectorT& GetObsHistory(size_t size) {
    if (size >= buffer_size_) {
      throw std::out_of_range(
          "Index out of range. HistoryBuffer.GetObsHistory(size_t)");
    }
    obs_short_history_ = MapT<VectorT>(
        data_ + ((current_index_ + 1) % buffer_size_) * single_obs_size_,
        size * single_obs_size_);
    return obs_short_history_;
  }

  /**
   * @brief Get full history
   *
   * @return Obs history
   */
  VectorT& GetObsHistory() {
    obs_history_ = MapT<VectorT>(
        data_ + ((current_index_ + 1) % buffer_size_) * single_obs_size_,
        single_obs_size_ * buffer_size_);
    return obs_history_;
  }

  void Reset() {
    current_index_ = -1;
    std::fill(data_, data_ + single_obs_size_ * (buffer_size_ * 2 - 1), 0);
  }

 private:
  T* data_;
  size_t single_obs_size_;
  size_t buffer_size_;
  size_t current_index_;

  VectorT obs_short_history_;
  VectorT obs_history_;
};

}  // namespace ovinf

#endif  // !HISTORY_BUFFER_HPP
