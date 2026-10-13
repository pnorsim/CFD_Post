#include "cfdpost/mmap_file.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "cfdpost/common.hpp"

namespace cfdpost {

MappedFile::MappedFile(const std::string& path) : path_(path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) fail("cannot open " + path + ": " + std::strerror(errno));
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    fail("cannot stat " + path);
  }
  size_ = static_cast<std::size_t>(st.st_size);
  if (size_ > 0) {
    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
      ::close(fd);
      fail("cannot mmap " + path + ": " + std::strerror(errno));
    }
    data_ = static_cast<const char*>(p);
  }
  ::close(fd);
}

MappedFile::~MappedFile() { release(); }

MappedFile::MappedFile(MappedFile&& o) noexcept
    : path_(std::move(o.path_)), data_(o.data_), size_(o.size_) {
  o.data_ = nullptr;
  o.size_ = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& o) noexcept {
  if (this != &o) {
    release();
    path_ = std::move(o.path_);
    data_ = o.data_;
    size_ = o.size_;
    o.data_ = nullptr;
    o.size_ = 0;
  }
  return *this;
}

void MappedFile::release() {
  if (data_) ::munmap(const_cast<char*>(data_), size_);
  data_ = nullptr;
  size_ = 0;
}

void MappedFile::advise_sequential(std::size_t offset, std::size_t len) const {
  if (!data_ || len == 0) return;
  long page = ::sysconf(_SC_PAGESIZE);
  std::size_t start = offset - offset % static_cast<std::size_t>(page);
  ::madvise(const_cast<char*>(data_) + start, len + (offset - start), MADV_WILLNEED);
}

}  // namespace cfdpost
