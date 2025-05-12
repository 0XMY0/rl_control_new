#ifndef CSV_LOGGER_HPP
#define CSV_LOGGER_HPP

#include <readerwriterqueue/readerwriterqueue.h>

#include <atomic>
#include <format>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

namespace ovinf {

class CsvLogger {
 public:
  using Number = std::variant<uint8_t, uint16_t, uint32_t, float, int64_t,
                              uint64_t, double>;
  using Ptr = std::shared_ptr<CsvLogger>;

  CsvLogger(std::string filename, std::vector<std::string> headers)
      : file_(filename, std::ios::out), data_queue_(1024) {
    if (file_.bad()) {
      file_bad_ = true;
      std::cerr << "Failed to create " << filename << std::endl;
    } else {
      file_bad_ = false;
      headers_size_ = headers.size();
      WriteTitle(headers);

      run_.store(true);
      write_thread_ = std::thread(&CsvLogger::WriteFile, this);
    }
  }

  ~CsvLogger() {
    run_.store(false);
    write_thread_.join();
    if (file_.is_open()) file_.close();
  }

  void Write(std::vector<Number> const& data) {
    if (file_bad_) return;
    if (data.size() != headers_size_) {
      if (data.size() > headers_size_) {
        std::cerr << "Data size is larger than header size" << std::endl;
      } else {
        std::cerr << "Data size is smaller than header size" << std::endl;
      }
    }

    data_queue_.enqueue(data);
  }

 private:
  /**
   * @brief Thread funtcion
   *
   */
  void WriteFile() {
    std::vector<Number> temp;

    while (run_) {
      while (data_queue_.try_dequeue(temp)) {
        std::string str = DataToCSV(temp);
        file_.write(str.data(), str.size());
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    while (data_queue_.try_dequeue(temp)) {
      std::string str = DataToCSV(temp);
      file_.write(str.data(), str.size());
    }
  }

  void WriteTitle(std::vector<std::string> headers) {
    std::string title;

    std::for_each(headers.begin(), headers.end() - 1,
                  [&title](std::string& header) {
                    std::format_to(std::back_inserter(title), "{},", header);
                  });
    std::format_to(std::back_inserter(title), "{}\n", headers.back());

    file_.write(title.data(), title.size());
  }

  std::string DataToCSV(std::vector<Number> data) {
    static NumberVisitor visitor;
    std::string csv_str;

    std::for_each(data.begin(), data.end() - 1, [&csv_str](Number& number) {
      std::format_to(std::back_inserter(csv_str), "{},",
                     std::visit(visitor, number));
    });
    std::format_to(std::back_inserter(csv_str), "{}\n",
                   std::visit(visitor, data.back()));

    return csv_str;
  }

  struct NumberVisitor {
    std::string operator()(double& n) { return std::format("{:.6f}", n); }
    template <class T>
    std::string operator()(T& n) {
      return std::format("{}", n);
    }
  };

 private:
  std::ofstream file_;
  moodycamel::ReaderWriterQueue<std::vector<Number>> data_queue_;
  std::thread write_thread_;

  bool file_bad_ = true;
  std::atomic_bool run_ = true;
  size_t headers_size_ = 0;
};

}  // namespace ovinf

#endif  // !CSV_LOGGER_HPP
