#ifndef OVINF_TRAITS_H
#define OVINF_TRAITS_H

#include <Eigen/Core>
#include <type_traits>

namespace ovinf {

// Eigen matrix type trait
template <typename T>
struct is_eigen_matrix : std::false_type {};

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
struct is_eigen_matrix<
    Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_eigen_matrix_v = is_eigen_matrix<T>::value;

// Eigen vector type trait
template <typename T>
struct is_eigen_vector : std::false_type {};

template <typename Scalar, int Rows, int Options, int MaxRows>
struct is_eigen_vector<Eigen::Matrix<Scalar, Rows, 1, Options, MaxRows, 1>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_eigen_vector_v = is_eigen_vector<T>::value;

}  // namespace ovinf

#endif  // !OVINF_TRAITS_H
