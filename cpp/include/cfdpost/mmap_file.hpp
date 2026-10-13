// Read-only memory-mapped file. Pages are shared between processes on a node,
// so many MPI ranks mapping the same solution file cost one copy in RAM.
#pragma once

#include <cstddef>
#include <string>

namespace cfdpost {

class MappedFile {
 public:
  MappedFile() = default;
  explicit MappedFile(const std::string& path);
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& o) noexcept;
  MappedFile& operator=(MappedFile&& o) noexcept;

  const char* data() const { return data_; }
  std::size_t size() const { return size_; }
  const std::string& path() const { return path_; }

  // Hint the kernel that [offset, offset+len) will be read sequentially.
  void advise_sequential(std::size_t offset, std::size_t len) const;

 private:
  void release();
  std::string path_;
  const char* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace cfdpost
