#pragma once
#include <cstdint>
#include <ostream>
#include <vector>

namespace Color {
class OstreamTag {};

class ColorfulTag : public OstreamTag {
public:
  int v_;

  constexpr explicit ColorfulTag(int v) : v_(v) {}
};

constexpr ColorfulTag kRed(31);
constexpr ColorfulTag kGreen(32);
constexpr ColorfulTag kYellow(33);
constexpr ColorfulTag kBlue(34);
constexpr ColorfulTag kMagenta(35);
constexpr ColorfulTag kCyan(36);
constexpr ColorfulTag kWhite(37);
constexpr ColorfulTag kGray(90);
constexpr ColorfulTag kLightRed(91);
constexpr ColorfulTag kLightGreen(92);
constexpr ColorfulTag kLightYellow(93);
constexpr ColorfulTag kLightBlue(94);
constexpr ColorfulTag kLightMagenta(95);
constexpr ColorfulTag kLightCyan(96);
constexpr ColorfulTag kLightWhite(97);
constexpr ColorfulTag kReset(39);

constexpr ColorfulTag kBgRed(41);
constexpr ColorfulTag kBgGreen(42);
constexpr ColorfulTag kBgYellow(43);
constexpr ColorfulTag kBgBlue(44);
constexpr ColorfulTag kBgMagenta(45);
constexpr ColorfulTag kBgCyan(46);
constexpr ColorfulTag kBgWhite(47);
constexpr ColorfulTag kBgGray(100);
constexpr ColorfulTag kBgLightRed(101);
constexpr ColorfulTag kBgLightGreen(102);
constexpr ColorfulTag kBgLightYellow(103);
constexpr ColorfulTag kBgLightBlue(104);
constexpr ColorfulTag kBgLightMagenta(105);
constexpr ColorfulTag kBgLightCyan(106);
constexpr ColorfulTag kBgLightWhite(107);
constexpr ColorfulTag kBgReset(49);

template <typename OsBase> class Ostream {
public:
  Ostream(OsBase &o) : o_(o) {}

  Ostream &operator<<(std::basic_ostream<char, std::char_traits<char>> &(*fp)(
      std::basic_ostream<char, std::char_traits<char>> &)) {
    o_ << fp;
    return *this;
  }

  template <typename T> Ostream &operator<<(T x) {
    ApplyAllColorfulTags();
    o_ << x;
    ApplyColorfulTag(kReset);
    ApplyColorfulTag(kBgReset);
    return *this;
  }

  Ostream &operator<<(const ColorfulTag &tag) {
    AddColorfulTag(tag);
    return *this;
  }

  void AddColorfulTag(const ColorfulTag &tag) { tags_.emplace_back(tag); }

private:
  void ApplyColorfulTag(const ColorfulTag &tag) {
    o_ << "\x1b[" << tag.v_ << "m";
  }

  void ApplyAllColorfulTags() {
    for (const auto &tag : tags_)
      ApplyColorfulTag(tag);
  }

  OsBase &o_;
  std::vector<ColorfulTag> tags_;
};

template <typename T, typename Tag,
          typename = std::enable_if_t<std::is_base_of_v<OstreamTag, Tag>>>
Ostream<T> operator<<(T &o, const Tag &x) {
  Ostream<T> rs(o);
  return rs << x;
}
} // namespace Color