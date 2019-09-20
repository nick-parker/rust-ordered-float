#![no_std]
#![cfg_attr(test, deny(warnings))]
#![deny(missing_docs)]

//! Wrappers for total order on Floats.

extern crate num_traits;
#[cfg(feature = "std")] extern crate std;


extern crate alga;
use alga::general::{RealField, AbstractMagma, Additive, Multiplicative, Identity,
    TwoSidedInverse};

extern crate alga_derive;
use alga_derive::Alga;


use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Rem,
               RemAssign, Sub, SubAssign};
use core::hash::{Hash, Hasher};
use core::fmt;
use core::mem;
use core::hint::unreachable_unchecked;
use core::str::FromStr;

use num_traits::{Bounded, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero};
#[cfg(feature = "std")]
use num_traits::Float;
#[cfg(not(feature = "std"))]
use num_traits::float::FloatCore as Float;

/// A wrapper around Floats providing an implementation of Ord and Hash.
///
/// A NaN value cannot be stored in this type.
#[deprecated(since = "0.6.0", note = "renamed to `NotNan`")]
pub type NotNaN<T> = NotNan<T>;

/// An error indicating an attempt to construct NotNan from a NaN
#[deprecated(since = "0.6.0", note = "renamed to `FloatIsNan`")]
pub type FloatIsNaN = FloatIsNan;

// masks for the parts of the IEEE 754 float
const SIGN_MASK: u64 = 0x8000000000000000u64;
const EXP_MASK: u64 = 0x7ff0000000000000u64;
const MAN_MASK: u64 = 0x000fffffffffffffu64;

// canonical raw bit patterns (for hashing)
const CANONICAL_NAN_BITS: u64 = 0x7ff8000000000000u64;
const CANONICAL_ZERO_BITS: u64 = 0x0u64;

#[allow(missing_docs)]
pub trait RealFloat: RealField + Float {
    // fn is_nan(self) -> bool;
    // fn nan() -> Self;
    // fn integer_decode(self) -> (u64, i16, i8);
}

impl RealFloat for f32 {
    // fn is_nan(self) -> bool {
    //     self != self
    // }
    // fn nan() -> f32 {
    //     std::f32::NAN
    // }
    // fn integer_decode(self) -> (u64, i16, i8) {
    //     let bits: u32 = unsafe { mem::transmute(self) };
    //     let sign: i8 = if bits >> 31 == 0 { 1 } else { -1 };
    //     let mut exponent: i16 = ((bits >> 23) & 0xff) as i16;
    //     let mantissa = if exponent == 0 {
    //         (bits & 0x7fffff) << 1
    //     } else {
    //         (bits & 0x7fffff) | 0x800000
    //     };
    //     // Exponent bias + mantissa shift
    //     exponent -= 127 + 23;
    //     (mantissa as u64, exponent, sign)
    // }
}

impl RealFloat for f64 {
    // fn is_nan(self) -> bool {
    //     self != self
    // }
    // fn nan() -> f64 {
    //     std::f64::NAN
    // }
    // fn integer_decode(self) -> (u64, i16, i8) {
    //     let bits: u64 = unsafe { mem::transmute(self) };
    //     let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    //     let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    //     let mantissa = if exponent == 0 {
    //         (bits & 0xfffffffffffff) << 1
    //     } else {
    //         (bits & 0xfffffffffffff) | 0x10000000000000
    //     };
    //     // Exponent bias + mantissa shift
    //     exponent -= 1023 + 52;
    //     (mantissa, exponent, sign)
    // }
}


/// A wrapper around Floats providing an implementation of Ord and Hash.
///
/// NaN is sorted as *greater* than all other values and *equal*
/// to itself, in contradiction with the IEEE standard.
#[derive(Debug, Default, Clone, Copy)]
#[repr(transparent)]
pub struct OrderedFloat<T: RealFloat>(pub T);

impl<T: RealFloat> OrderedFloat<T> {
    /// Get the value out.
    pub fn into_inner(self) -> T {
        let OrderedFloat(val) = self;
        val
    }
}

impl<T: RealFloat> AsRef<T> for OrderedFloat<T> {
    fn as_ref(&self) -> &T {
        let OrderedFloat(ref val) = *self;
        val
    }
}

impl<T: RealFloat> AsMut<T> for OrderedFloat<T> {
    fn as_mut(&mut self) -> &mut T {
        let OrderedFloat(ref mut val) = *self;
        val
    }
}

impl<T: RealFloat> PartialOrd for OrderedFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: RealFloat> Ord for OrderedFloat<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        let lhs = self.as_ref();
        let rhs = other.as_ref();
        match lhs.partial_cmp(&rhs) {
            Some(ordering) => ordering,
            None => {
                if lhs.is_nan() {
                    if rhs.is_nan() {
                        Ordering::Equal
                    } else {
                        Ordering::Greater
                    }
                } else {
                    Ordering::Less
                }
            }
        }
    }
}

impl<T: RealFloat> PartialEq for OrderedFloat<T> {
    fn eq(&self, other: &OrderedFloat<T>) -> bool {
        if self.as_ref().is_nan() {
            other.as_ref().is_nan()
        } else {
            self.as_ref() == other.as_ref()
        }
    }
}

impl<T: RealFloat> Hash for OrderedFloat<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if self.is_nan() {
            // normalize to one representation of NaN
            hash_float(&T::nan(), state)
        } else {
            hash_float(self.as_ref(), state)
        }
    }
}

impl<T: RealFloat + fmt::Display> fmt::Display for OrderedFloat<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        core::fmt::Display::fmt(&self.as_ref(), f)
    }
}

impl Into<f32> for OrderedFloat<f32> {
    fn into(self) -> f32 {
        self.into_inner()
    }
}

impl Into<f64> for OrderedFloat<f64> {
    fn into(self) -> f64 {
        self.into_inner()
    }
}

impl<T: RealFloat> From<T> for OrderedFloat<T> {
    fn from(val: T) -> Self {
        OrderedFloat(val)
    }
}

impl<T: RealFloat> Deref for OrderedFloat<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T: RealFloat> DerefMut for OrderedFloat<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

impl<T: RealFloat> Eq for OrderedFloat<T> {}

impl<T: RealFloat> Bounded for OrderedFloat<T> {
    fn min_value() -> Self {
        OrderedFloat(num_traits::Float::min_value())
    }

    fn max_value() -> Self {
        OrderedFloat(num_traits::Float::max_value())
    }
}

impl<T: RealFloat + FromStr> FromStr for OrderedFloat<T> {
    type Err = T::Err;

    /// Convert a &str to `OrderedFloat`. Returns an error if the string fails to parse.
    ///
    /// ```
    /// use ordered_float::OrderedFloat;
    ///
    /// assert!("-10".parse::<OrderedFloat<f32>>().is_ok());
    /// assert!("abc".parse::<OrderedFloat<f32>>().is_err());
    /// assert!("NaN".parse::<OrderedFloat<f32>>().is_ok());
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        T::from_str(s).map(OrderedFloat)
    }
}

impl<T: RealFloat> Neg for OrderedFloat<T> {
    type Output = Self;

    fn neg(self) -> Self {
        OrderedFloat(-self.0)
    }
}

/// A wrapper around Floats providing an implementation of Ord and Hash.
///
/// A NaN value cannot be stored in this type.
#[derive(PartialOrd, PartialEq, Debug, Default, Clone, Copy, Alga)]
#[alga_traits(Ring(Additive, Multiplicative))]
#[repr(transparent)]
pub struct NotNan<T: RealFloat>(T);

impl<T: RealFloat> NotNan<T> {
    /// Create a NotNan value.
    ///
    /// Returns Err if val is NaN
    pub fn new(val: T) -> Result<Self, FloatIsNan> {
        match val {
            ref val if val.is_nan() => Err(FloatIsNan),
            val => Ok(NotNan(val)),
        }
    }

    /// Create a NotNan value from a value that is guaranteed to not be NaN
    ///
    /// Behaviour is undefined if `val` is NaN
    pub unsafe fn unchecked_new(val: T) -> Self {
        debug_assert!(!val.is_nan());
        NotNan(val)
    }

    /// Get the value out.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T: RealFloat> AsRef<T> for NotNan<T> {
    fn as_ref(&self) -> &T {
        &self.0
    }
}

impl<T: RealFloat> Ord for NotNan<T> {
    fn cmp(&self, other: &NotNan<T>) -> Ordering {
        match self.partial_cmp(&other) {
            Some(ord) => ord,
            None => unsafe { unreachable_unchecked() },
        }
    }
}

impl<T: RealFloat> Hash for NotNan<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_float(self.as_ref(), state)
    }
}

impl<T: RealFloat + fmt::Display> fmt::Display for NotNan<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        core::fmt::Display::fmt(&self.as_ref(), f)
    }
}

impl From<NotNan<f32>> for f32 {
    fn from(value: NotNan<f32>) -> Self {
        value.into_inner()
    }
}

impl From<NotNan<f64>> for f64 {
    fn from(value: NotNan<f64>) -> Self {
        value.into_inner()
    }
}

/// Creates a NotNan value from a Float.
///
/// Panics if the provided value is NaN or the computation results in NaN
impl<T: RealFloat> From<T> for NotNan<T> {
    fn from(v: T) -> Self {
        NotNan::new(v).expect("Tried to create a NotNan from a NaN")
    }
}

impl<T: RealFloat> Deref for NotNan<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T: RealFloat + PartialEq> Eq for NotNan<T> {}

/// Adds two NotNans.
///
/// Panics if the computation results in NaN
impl<T: RealFloat> Add for NotNan<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self + other.0
    }
}

/// Adds a float directly.
///
/// Panics if the provided value is NaN or the computation results in NaN
impl<T: RealFloat> Add<T> for NotNan<T> {
    type Output = Self;

    fn add(self, other: T) -> Self {
        NotNan::new(self.0 + other).expect("Addition resulted in NaN")
    }
}

impl<T: RealFloat + AddAssign> AddAssign for NotNan<T> {
    fn add_assign(&mut self, other: Self) {
        *self += other.0;
    }
}

/// Adds a float directly.
///
/// Panics if the provided value is NaN.
impl<T: RealFloat + AddAssign> AddAssign<T> for NotNan<T> {
    fn add_assign(&mut self, other: T) {
        self.0 += other;
        assert!(!self.0.is_nan(), "Addition resulted in NaN");
    }
}

impl<T: RealFloat> Sub for NotNan<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self - other.0
    }
}

/// Subtracts a float directly.
///
/// Panics if the provided value is NaN or the computation results in NaN
impl<T: RealFloat> Sub<T> for NotNan<T> {
    type Output = Self;

    fn sub(self, other: T) -> Self {
        NotNan::new(self.0 - other).expect("Subtraction resulted in NaN")
    }
}

impl<T: RealFloat + SubAssign> SubAssign for NotNan<T> {
    fn sub_assign(&mut self, other: Self) {
        *self -= other.0
    }
}

/// Subtracts a float directly.
///
/// Panics if the provided value is NaN or the computation results in NaN
impl<T: RealFloat + SubAssign> SubAssign<T> for NotNan<T> {
    fn sub_assign(&mut self, other: T) {
        self.0 -= other;
        assert!(!self.0.is_nan(), "Subtraction resulted in NaN");
    }
}

impl<T: RealFloat> Mul for NotNan<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self * other.0
    }
}

/// Multiplies a float directly.
///
/// Panics if the provided value is NaN or the computation results in NaN
impl<T: RealFloat> Mul<T> for NotNan<T> {
    type Output = Self;

    fn mul(self, other: T) -> Self {
        NotNan::new(self.0 * other).expect("Multiplication resulted in NaN")
    }
}

impl<T: RealFloat + MulAssign> MulAssign for NotNan<T> {
    fn mul_assign(&mut self, other: Self) {
        *self *= other.0
    }
}

/// Multiplies a float directly.
///
/// Panics if the provided value is NaN.
impl<T: RealFloat + MulAssign> MulAssign<T> for NotNan<T> {
    fn mul_assign(&mut self, other: T) {
        self.0 *= other;
        assert!(!self.0.is_nan(), "Multiplication resulted in NaN");
    }
}

impl<T: RealFloat> Div for NotNan<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        self / other.0
    }
}

/// Divides a float directly.
///
/// Panics if the provided value is NaN or the computation results in NaN
impl<T: RealFloat> Div<T> for NotNan<T> {
    type Output = Self;

    fn div(self, other: T) -> Self {
        NotNan::new(self.0 / other).expect("Division resulted in NaN")
    }
}

impl<T: RealFloat + DivAssign> DivAssign for NotNan<T> {
    fn div_assign(&mut self, other: Self) {
        *self /= other.0;
    }
}

/// Divides a float directly.
///
/// Panics if the provided value is NaN or the computation results in NaN
impl<T: RealFloat + DivAssign> DivAssign<T> for NotNan<T> {
    fn div_assign(&mut self, other: T) {
        self.0 /= other;
        assert!(!self.0.is_nan(), "Division resulted in NaN");
    }
}

impl<T: RealFloat> Rem for NotNan<T> {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        self % other.0
    }
}

/// Calculates `%` with a float directly.
///
/// Panics if the provided value is NaN or the computation results in NaN
impl<T: RealFloat> Rem<T> for NotNan<T> {
    type Output = Self;

    fn rem(self, other: T) -> Self {
        NotNan::new(self.0 % other).expect("Rem resulted in NaN")
    }
}

impl<T: RealFloat + RemAssign> RemAssign for NotNan<T> {
    fn rem_assign(&mut self, other: Self) {
        *self %= other.0
    }
}

/// Calculates `%=` with a float directly.
///
/// Panics if the provided value is NaN or the computation results in NaN
impl<T: RealFloat + RemAssign> RemAssign<T> for NotNan<T> {
    fn rem_assign(&mut self, other: T) {
        self.0 %= other;
        assert!(!self.0.is_nan(), "Rem resulted in NaN");
    }
}

impl<T: RealFloat> Neg for NotNan<T> {
    type Output = Self;

    fn neg(self) -> Self {
        NotNan(-self.0)
    }
}

/// An error indicating an attempt to construct NotNan from a NaN
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct FloatIsNan;

#[cfg(feature = "std")]
impl std::error::Error for FloatIsNan {
    fn description(&self) -> &str {
        "NotNan constructed with NaN"
    }
}

impl fmt::Display for FloatIsNan {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NotNan constructed with NaN")
    }
}

#[cfg(feature = "std")]
impl Into<std::io::Error> for FloatIsNan {
    fn into(self) -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, self)
    }
}

#[inline]
fn hash_float<F: RealFloat, H: Hasher>(f: &F, state: &mut H) {
    raw_double_bits(f).hash(state);
}

#[inline]
fn raw_double_bits<F: RealFloat>(f: &F) -> u64 {
    if f.is_nan() {
        return CANONICAL_NAN_BITS;
    }

    let (man, exp, sign) = f.integer_decode();
    if man == 0 {
        return CANONICAL_ZERO_BITS;
    }

    let exp_u64 = unsafe { mem::transmute::<i16, u16>(exp) } as u64;
    let sign_u64 = if sign > 0 { 1u64 } else { 0u64 };
    (man & MAN_MASK) | ((exp_u64 << 52) & EXP_MASK) | ((sign_u64 << 63) & SIGN_MASK)
}

impl<T: RealFloat> Zero for NotNan<T> {
    fn zero() -> Self { NotNan(T::zero()) }

    fn is_zero(&self) -> bool { self.0.is_zero() }
}

impl<T: RealFloat> One for NotNan<T> {
    fn one() -> Self { NotNan(T::one()) }
}

impl<T: RealFloat> Bounded for NotNan<T> {
    fn min_value() -> Self {
        NotNan(num_traits::Float::min_value())
    }

    fn max_value() -> Self {
        NotNan(num_traits::Float::max_value())
    }
}

impl<T: RealFloat + FromStr> FromStr for NotNan<T> {
    type Err = ParseNotNanError<T::Err>;

    /// Convert a &str to `NotNan`. Returns an error if the string fails to parse,
    /// or if the resulting value is NaN
    ///
    /// ```
    /// use ordered_float::NotNan;
    ///
    /// assert!("-10".parse::<NotNan<f32>>().is_ok());
    /// assert!("abc".parse::<NotNan<f32>>().is_err());
    /// assert!("NaN".parse::<NotNan<f32>>().is_err());
    /// ```
    fn from_str(src: &str) -> Result<Self, Self::Err> {
        src.parse()
            .map_err(ParseNotNanError::ParseFloatError)
            .and_then(|f| NotNan::new(f).map_err(|_| ParseNotNanError::IsNaN))
    }
}

impl<T: RealFloat + FromPrimitive> FromPrimitive for NotNan<T> {
    fn from_i64(n: i64) -> Option<Self> { T::from_i64(n).and_then(|n| NotNan::new(n).ok()) }
    fn from_u64(n: u64) -> Option<Self> { T::from_u64(n).and_then(|n| NotNan::new(n).ok()) }

    fn from_isize(n: isize) -> Option<Self> { T::from_isize(n).and_then(|n| NotNan::new(n).ok()) }
    fn from_i8(n: i8) -> Option<Self> { T::from_i8(n).and_then(|n| NotNan::new(n).ok()) }
    fn from_i16(n: i16) -> Option<Self> { T::from_i16(n).and_then(|n| NotNan::new(n).ok()) }
    fn from_i32(n: i32) -> Option<Self> { T::from_i32(n).and_then(|n| NotNan::new(n).ok()) }
    fn from_usize(n: usize) -> Option<Self> { T::from_usize(n).and_then(|n| NotNan::new(n).ok()) }
    fn from_u8(n: u8) -> Option<Self> { T::from_u8(n).and_then(|n| NotNan::new(n).ok()) }
    fn from_u16(n: u16) -> Option<Self> { T::from_u16(n).and_then(|n| NotNan::new(n).ok()) }
    fn from_u32(n: u32) -> Option<Self> { T::from_u32(n).and_then(|n| NotNan::new(n).ok()) }
    fn from_f32(n: f32) -> Option<Self> { T::from_f32(n).and_then(|n| NotNan::new(n).ok()) }
    fn from_f64(n: f64) -> Option<Self> { T::from_f64(n).and_then(|n| NotNan::new(n).ok()) }
}

impl<T: RealFloat> ToPrimitive for NotNan<T> {
    fn to_i64(&self) -> Option<i64> { self.0.to_i64() }
    fn to_u64(&self) -> Option<u64> { self.0.to_u64() }

    fn to_isize(&self) -> Option<isize> { self.0.to_isize() }
    fn to_i8(&self) -> Option<i8> { self.0.to_i8() }
    fn to_i16(&self) -> Option<i16> { self.0.to_i16() }
    fn to_i32(&self) -> Option<i32> { self.0.to_i32() }
    fn to_usize(&self) -> Option<usize> { self.0.to_usize() }
    fn to_u8(&self) -> Option<u8> { self.0.to_u8() }
    fn to_u16(&self) -> Option<u16> { self.0.to_u16() }
    fn to_u32(&self) -> Option<u32> { self.0.to_u32() }
    fn to_f32(&self) -> Option<f32> { self.0.to_f32() }
    fn to_f64(&self) -> Option<f64> { self.0.to_f64() }
}

/// An error indicating a parse error from a string for `NotNan`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ParseNotNanError<E> {
    /// A plain parse error from the underlying float type.
    ParseFloatError(E),
    /// The parsed float value resulted in a NaN.
    IsNaN,
}

#[cfg(feature = "std")]
impl<E: fmt::Debug> std::error::Error for ParseNotNanError<E> {
    fn description(&self) -> &str {
        return "Error parsing a not-NaN floating point value";
    }

    // TODO: add an implementation of cause(). This will be breaking because it requires E: Error.
}

impl<E: fmt::Debug> fmt::Display for ParseNotNanError<E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO: replace this with a human readable fmt. Will require E: Display.
        <Self as fmt::Debug>::fmt(self, f)
    }
}

impl<T: RealFloat> Num for NotNan<T> {
    type FromStrRadixErr = ParseNotNanError<T::FromStrRadixErr>;

    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(src, radix)
            .map_err(|err| ParseNotNanError::ParseFloatError(err))
            .and_then(|n| NotNan::new(n).map_err(|_| ParseNotNanError::IsNaN))
    }
}

impl<T: RealFloat + Signed> Signed for NotNan<T> {
    fn abs(&self) -> Self { NotNan(num_traits::Float::abs(self.0)) }

    fn abs_sub(&self, other: &Self) -> Self {
        NotNan::new(Signed::abs_sub(&self.0, &other.0)).expect("Subtraction resulted in NaN")
    }

    fn signum(&self) -> Self { NotNan(num_traits::Float::signum(self.0)) }
    fn is_positive(&self) -> bool { self.0.is_positive() }
    fn is_negative(&self) -> bool { self.0.is_negative() }
}

impl<T: RealFloat> NumCast for NotNan<T> {
    fn from<F: ToPrimitive>(n: F) -> Option<Self> {
        T::from(n).and_then(|n| NotNan::new(n).ok())
    }
}

impl<T: RealFloat + AbstractMagma<Additive>> AbstractMagma<Additive> for NotNan<T> {
    fn operate(&self, right: &Self) -> Self {
        NotNan(AbstractMagma::<Additive>::operate(&self.0, &right.0))
    }
}

impl<T: RealFloat +  TwoSidedInverse<Additive>> TwoSidedInverse<Additive> for NotNan<T> {
    fn two_sided_inverse(&self) -> Self {
        NotNan(TwoSidedInverse::<Additive>::two_sided_inverse(&self.0))
    }
}

impl<T: RealFloat +  TwoSidedInverse<Multiplicative>> TwoSidedInverse<Multiplicative> for NotNan<T> {
    fn two_sided_inverse(&self) -> Self {
        NotNan(TwoSidedInverse::<Multiplicative>::two_sided_inverse(&self.0))
    }
}

impl<T: RealFloat +  Identity<Additive>> Identity<Additive> for NotNan<T> {
    fn identity() -> Self {
        NotNan(Identity::<Additive>::identity())
    }
}

impl<T: RealFloat +  AbstractMagma<Multiplicative>> AbstractMagma<Multiplicative> for NotNan<T> {
    fn operate(&self, right: &Self) -> Self {
        NotNan(AbstractMagma::<Multiplicative>::operate(&self.0, &right.0))
    }
}

impl<T: RealFloat +  Identity<Multiplicative>> Identity<Multiplicative> for NotNan<T> {
    fn identity() -> Self {
        NotNan(Identity::<Multiplicative>::identity())
    }
}

// alga::impl_ring!(<Additive, Multiplicative> for NotNan<T> where T: AbstractRing);

#[cfg(feature = "serde")]
mod impl_serde {
    extern crate serde;
    use self::serde::{Serialize, Serializer, Deserialize, Deserializer};
    use self::serde::de::{Error, Unexpected};
    use super::{OrderedFloat, NotNan};
    #[cfg(feature = "std")]
    use num_traits::Float;
    #[cfg(not(feature = "std"))]
    use num_traits::float::FloatCore as Float;
    use core::f64;

    #[cfg(test)]
    extern crate serde_test;
    #[cfg(test)]
    use self::serde_test::{Token, assert_tokens, assert_de_tokens_error};

    impl<T: RealFloat + Serialize> Serialize for OrderedFloat<T> {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            self.0.serialize(s)
        }
    }

    impl<'de, T: RealFloat + Deserialize<'de>> Deserialize<'de> for OrderedFloat<T> {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            T::deserialize(d).map(OrderedFloat)
        }
    }

    impl<T: RealFloat + Serialize> Serialize for NotNan<T> {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            self.0.serialize(s)
        }
    }

    impl<'de, T: RealFloat + Deserialize<'de>> Deserialize<'de> for NotNan<T> {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            let float = T::deserialize(d)?;
            NotNan::new(float).map_err(|_| {
                Error::invalid_value(Unexpected::Float(f64::NAN), &"float (but not NaN)")
            })
        }
    }

    #[test]
    fn test_ordered_float() {
        let float = OrderedFloat(1.0f64);
        assert_tokens(&float, &[Token::F64(1.0)]);
    }

    #[test]
    fn test_not_nan() {
        let float = NotNan(1.0f64);
        assert_tokens(&float, &[Token::F64(1.0)]);
    }

    #[test]
    fn test_fail_on_nan() {
        assert_de_tokens_error::<NotNan<f64>>(
            &[Token::F64(f64::NAN)],
            "invalid value: floating point `NaN`, expected float (but not NaN)");
    }
}
