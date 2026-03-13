package shard

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Column encoding types
const (
	ColEncPlain           byte = 0x00
	ColEncDictionary      byte = 0x01
	ColEncRLE             byte = 0x02
	ColEncDelta           byte = 0x03
	ColEncByteStreamSplit byte = 0x04
)

// ColType represents a column data type.
type ColType byte

const (
	ColTypeBool     ColType = 0x01
	ColTypeInt8     ColType = 0x02
	ColTypeInt16    ColType = 0x03
	ColTypeInt32    ColType = 0x04
	ColTypeInt64    ColType = 0x05
	ColTypeUint8    ColType = 0x06
	ColTypeUint16   ColType = 0x07
	ColTypeUint32   ColType = 0x08
	ColTypeUint64   ColType = 0x09
	ColTypeFloat16  ColType = 0x0A
	ColTypeFloat32  ColType = 0x0B
	ColTypeFloat64  ColType = 0x0C
	ColTypeString   ColType = 0x0D
	ColTypeBytes    ColType = 0x0E
	ColTypeDatetime ColType = 0x0F
	ColTypeUUID     ColType = 0x10
	ColTypeBFloat16 ColType = 0x11
)

// Width returns the byte width of a fixed-width type, or -1 for variable-width.
func (t ColType) Width() int {
	switch t {
	case ColTypeBool, ColTypeInt8, ColTypeUint8:
		return 1
	case ColTypeInt16, ColTypeUint16, ColTypeFloat16, ColTypeBFloat16:
		return 2
	case ColTypeInt32, ColTypeUint32, ColTypeFloat32:
		return 4
	case ColTypeInt64, ColTypeUint64, ColTypeFloat64, ColTypeDatetime:
		return 8
	case ColTypeUUID:
		return 16
	case ColTypeString, ColTypeBytes:
		return -1
	default:
		return -1
	}
}

// String returns the human-readable name of the column type.
func (t ColType) String() string {
	switch t {
	case ColTypeBool:
		return "bool"
	case ColTypeInt8:
		return "int8"
	case ColTypeInt16:
		return "int16"
	case ColTypeInt32:
		return "int32"
	case ColTypeInt64:
		return "int64"
	case ColTypeUint8:
		return "uint8"
	case ColTypeUint16:
		return "uint16"
	case ColTypeUint32:
		return "uint32"
	case ColTypeUint64:
		return "uint64"
	case ColTypeFloat16:
		return "float16"
	case ColTypeFloat32:
		return "float32"
	case ColTypeFloat64:
		return "float64"
	case ColTypeString:
		return "string"
	case ColTypeBytes:
		return "bytes"
	case ColTypeDatetime:
		return "datetime"
	case ColTypeUUID:
		return "uuid"
	case ColTypeBFloat16:
		return "bfloat16"
	default:
		return fmt.Sprintf("unknown(0x%02X)", byte(t))
	}
}

// ColumnChunk represents a decoded column chunk.
type ColumnChunk struct {
	Encoding byte
	Count    uint32
	Nullable bool
	Valid    []bool // validity bitmap, len == Count if nullable
	Data     []byte // raw encoded data (after header + bitmap)
}

// ---------------------------------------------------------------------------
// Null bitmap helpers
// ---------------------------------------------------------------------------

// PackNullBitmap packs a bool slice into a bitmap (1=valid, 0=null).
// Bit 0 of byte 0 = row 0.
func PackNullBitmap(valid []bool) []byte {
	n := len(valid)
	bitmapLen := (n + 7) / 8
	buf := make([]byte, bitmapLen)
	for i, v := range valid {
		if v {
			buf[i/8] |= 1 << uint(i%8)
		}
	}
	return buf
}

// UnpackNullBitmap unpacks a bitmap into a bool slice.
func UnpackNullBitmap(bitmap []byte, count int) []bool {
	valid := make([]bool, count)
	for i := 0; i < count; i++ {
		if bitmap[i/8]&(1<<uint(i%8)) != 0 {
			valid[i] = true
		}
	}
	return valid
}

// ---------------------------------------------------------------------------
// Column chunk marshal / unmarshal
// ---------------------------------------------------------------------------

// MarshalColumnChunk writes a complete column chunk blob:
// [encoding: 1 byte][count: u32 LE][null_bitmap if nullable][encoded_data]
func MarshalColumnChunk(encoding byte, count uint32, nullable bool, valid []bool, encodedData []byte) []byte {
	size := 1 + 4 + len(encodedData)
	if nullable {
		size += (int(count) + 7) / 8
	}
	buf := make([]byte, 0, size)
	buf = append(buf, encoding)
	tmp := make([]byte, 4)
	binary.LittleEndian.PutUint32(tmp, count)
	buf = append(buf, tmp...)
	if nullable {
		buf = append(buf, PackNullBitmap(valid)...)
	}
	buf = append(buf, encodedData...)
	return buf
}

// UnmarshalColumnChunk parses a column chunk blob header.
// Returns the ColumnChunk with Data pointing to the encoded payload.
func UnmarshalColumnChunk(data []byte, nullable bool) (*ColumnChunk, error) {
	if len(data) < 5 {
		return nil, fmt.Errorf("column chunk too short: %d bytes", len(data))
	}
	cc := &ColumnChunk{
		Encoding: data[0],
		Count:    binary.LittleEndian.Uint32(data[1:5]),
		Nullable: nullable,
	}
	offset := 5
	if nullable {
		bitmapLen := (int(cc.Count) + 7) / 8
		if len(data) < offset+bitmapLen {
			return nil, fmt.Errorf("column chunk too short for null bitmap")
		}
		cc.Valid = UnpackNullBitmap(data[offset:offset+bitmapLen], int(cc.Count))
		offset += bitmapLen
	}
	cc.Data = data[offset:]
	return cc, nil
}

// ---------------------------------------------------------------------------
// 1. PLAIN encoding
// ---------------------------------------------------------------------------

// EncodePlainInt32 encodes int32 values as concatenated little-endian bytes.
func EncodePlainInt32(values []int32) []byte {
	buf := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(v))
	}
	return buf
}

// DecodePlainInt32 decodes concatenated little-endian int32 values.
func DecodePlainInt32(data []byte, count int) []int32 {
	out := make([]int32, count)
	for i := 0; i < count; i++ {
		out[i] = int32(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out
}

// EncodePlainInt64 encodes int64 values as concatenated little-endian bytes.
func EncodePlainInt64(values []int64) []byte {
	buf := make([]byte, len(values)*8)
	for i, v := range values {
		binary.LittleEndian.PutUint64(buf[i*8:], uint64(v))
	}
	return buf
}

// DecodePlainInt64 decodes concatenated little-endian int64 values.
func DecodePlainInt64(data []byte, count int) []int64 {
	out := make([]int64, count)
	for i := 0; i < count; i++ {
		out[i] = int64(binary.LittleEndian.Uint64(data[i*8:]))
	}
	return out
}

// EncodePlainFloat32 encodes float32 values as concatenated little-endian bytes.
func EncodePlainFloat32(values []float32) []byte {
	buf := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// DecodePlainFloat32 decodes concatenated little-endian float32 values.
func DecodePlainFloat32(data []byte, count int) []float32 {
	out := make([]float32, count)
	for i := 0; i < count; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out
}

// EncodePlainFloat64 encodes float64 values as concatenated little-endian bytes.
func EncodePlainFloat64(values []float64) []byte {
	buf := make([]byte, len(values)*8)
	for i, v := range values {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
	}
	return buf
}

// DecodePlainFloat64 decodes concatenated little-endian float64 values.
func DecodePlainFloat64(data []byte, count int) []float64 {
	out := make([]float64, count)
	for i := 0; i < count; i++ {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*8:]))
	}
	return out
}

// EncodePlainUint64 encodes uint64 values as concatenated little-endian bytes.
func EncodePlainUint64(values []uint64) []byte {
	buf := make([]byte, len(values)*8)
	for i, v := range values {
		binary.LittleEndian.PutUint64(buf[i*8:], v)
	}
	return buf
}

// DecodePlainUint64 decodes concatenated little-endian uint64 values.
func DecodePlainUint64(data []byte, count int) []uint64 {
	out := make([]uint64, count)
	for i := 0; i < count; i++ {
		out[i] = binary.LittleEndian.Uint64(data[i*8:])
	}
	return out
}

// EncodePlainStrings encodes strings with uint32 LE length prefix per value.
func EncodePlainStrings(values []string) []byte {
	size := 0
	for _, s := range values {
		size += 4 + len(s)
	}
	buf := make([]byte, 0, size)
	tmp := make([]byte, 4)
	for _, s := range values {
		binary.LittleEndian.PutUint32(tmp, uint32(len(s)))
		buf = append(buf, tmp...)
		buf = append(buf, s...)
	}
	return buf
}

// DecodePlainStrings decodes length-prefixed strings.
func DecodePlainStrings(data []byte, count int) []string {
	out := make([]string, count)
	off := 0
	for i := 0; i < count; i++ {
		n := int(binary.LittleEndian.Uint32(data[off:]))
		off += 4
		out[i] = string(data[off : off+n])
		off += n
	}
	return out
}

// EncodePlainBytes encodes byte slices with uint32 LE length prefix per value.
func EncodePlainBytes(values [][]byte) []byte {
	size := 0
	for _, b := range values {
		size += 4 + len(b)
	}
	buf := make([]byte, 0, size)
	tmp := make([]byte, 4)
	for _, b := range values {
		binary.LittleEndian.PutUint32(tmp, uint32(len(b)))
		buf = append(buf, tmp...)
		buf = append(buf, b...)
	}
	return buf
}

// DecodePlainBytes decodes length-prefixed byte slices.
func DecodePlainBytes(data []byte, count int) [][]byte {
	out := make([][]byte, count)
	off := 0
	for i := 0; i < count; i++ {
		n := int(binary.LittleEndian.Uint32(data[off:]))
		off += 4
		out[i] = make([]byte, n)
		copy(out[i], data[off:off+n])
		off += n
	}
	return out
}

// ---------------------------------------------------------------------------
// 2. DICTIONARY encoding
// ---------------------------------------------------------------------------

// EncodeDictionary encodes strings using dictionary encoding.
// Format: [dict_size: uint32 LE][dictionary: PLAIN strings][indices: width × count]
// Index width: uint8 if dict_size <= 255, uint16 if <= 65535, uint32 otherwise.
func EncodeDictionary(values []string) []byte {
	// Build dictionary
	dictMap := make(map[string]uint32)
	var dict []string
	indices := make([]uint32, len(values))
	for i, v := range values {
		idx, ok := dictMap[v]
		if !ok {
			idx = uint32(len(dict))
			dictMap[v] = idx
			dict = append(dict, v)
		}
		indices[i] = idx
	}

	dictSize := uint32(len(dict))

	// Encode dictionary as PLAIN strings
	dictData := EncodePlainStrings(dict)

	// Determine index width
	var indexData []byte
	if dictSize <= 255 {
		indexData = make([]byte, len(values))
		for i, idx := range indices {
			indexData[i] = byte(idx)
		}
	} else if dictSize <= 65535 {
		indexData = make([]byte, len(values)*2)
		for i, idx := range indices {
			binary.LittleEndian.PutUint16(indexData[i*2:], uint16(idx))
		}
	} else {
		indexData = make([]byte, len(values)*4)
		for i, idx := range indices {
			binary.LittleEndian.PutUint32(indexData[i*4:], idx)
		}
	}

	buf := make([]byte, 4+len(dictData)+len(indexData))
	binary.LittleEndian.PutUint32(buf, dictSize)
	copy(buf[4:], dictData)
	copy(buf[4+len(dictData):], indexData)
	return buf
}

// DecodeDictionary decodes dictionary-encoded strings.
func DecodeDictionary(data []byte, count int) []string {
	dictSize := int(binary.LittleEndian.Uint32(data[:4]))
	off := 4

	// Decode dictionary (PLAIN strings)
	dict := DecodePlainStrings(data[off:], dictSize)

	// Advance past the dictionary data
	for i := 0; i < dictSize; i++ {
		n := int(binary.LittleEndian.Uint32(data[off:]))
		off += 4 + n
	}

	// Read indices
	out := make([]string, count)
	if dictSize <= 255 {
		for i := 0; i < count; i++ {
			out[i] = dict[data[off+i]]
		}
	} else if dictSize <= 65535 {
		for i := 0; i < count; i++ {
			idx := binary.LittleEndian.Uint16(data[off+i*2:])
			out[i] = dict[idx]
		}
	} else {
		for i := 0; i < count; i++ {
			idx := binary.LittleEndian.Uint32(data[off+i*4:])
			out[i] = dict[idx]
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// 3. RLE encoding
// ---------------------------------------------------------------------------

// EncodeRLEBool encodes booleans using run-length encoding.
// Format: [bit_width: 1 byte (always 1)][runs: (count: uvarint)(value: 1 byte)]
func EncodeRLEBool(values []bool) []byte {
	if len(values) == 0 {
		return []byte{1}
	}
	buf := []byte{1} // bit_width = 1
	uvarBuf := make([]byte, binary.MaxVarintLen64)

	i := 0
	for i < len(values) {
		cur := values[i]
		runLen := 1
		for i+runLen < len(values) && values[i+runLen] == cur {
			runLen++
		}
		n := binary.PutUvarint(uvarBuf, uint64(runLen))
		buf = append(buf, uvarBuf[:n]...)
		if cur {
			buf = append(buf, 1)
		} else {
			buf = append(buf, 0)
		}
		i += runLen
	}
	return buf
}

// DecodeRLEBool decodes run-length encoded booleans.
func DecodeRLEBool(data []byte, count int) []bool {
	out := make([]bool, 0, count)
	// data[0] is bit_width (1), skip it
	off := 1
	for len(out) < count {
		runLen, n := binary.Uvarint(data[off:])
		off += n
		val := data[off] != 0
		off++
		for j := uint64(0); j < runLen && len(out) < count; j++ {
			out = append(out, val)
		}
	}
	return out
}

// EncodeRLEInt32 encodes int32 values using run-length encoding.
// Format: [bit_width: 1 byte][runs: (count: uvarint)(value: bit_width bytes, byte-aligned)]
func EncodeRLEInt32(values []int32, bitWidth int) []byte {
	if len(values) == 0 {
		return []byte{byte(bitWidth)}
	}
	byteWidth := (bitWidth + 7) / 8
	buf := []byte{byte(bitWidth)}
	uvarBuf := make([]byte, binary.MaxVarintLen64)
	valBuf := make([]byte, 4)

	i := 0
	for i < len(values) {
		cur := values[i]
		runLen := 1
		for i+runLen < len(values) && values[i+runLen] == cur {
			runLen++
		}
		n := binary.PutUvarint(uvarBuf, uint64(runLen))
		buf = append(buf, uvarBuf[:n]...)
		binary.LittleEndian.PutUint32(valBuf, uint32(cur))
		buf = append(buf, valBuf[:byteWidth]...)
		i += runLen
	}
	return buf
}

// DecodeRLEInt32 decodes run-length encoded int32 values.
func DecodeRLEInt32(data []byte, count int) []int32 {
	out := make([]int32, 0, count)
	bitWidth := int(data[0])
	byteWidth := (bitWidth + 7) / 8
	off := 1
	for len(out) < count {
		runLen, n := binary.Uvarint(data[off:])
		off += n
		// Read value from byteWidth bytes, zero-extended to 4 bytes
		var valBytes [4]byte
		copy(valBytes[:], data[off:off+byteWidth])
		val := int32(binary.LittleEndian.Uint32(valBytes[:]))
		off += byteWidth
		for j := uint64(0); j < runLen && len(out) < count; j++ {
			out = append(out, val)
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// 4. DELTA encoding
// ---------------------------------------------------------------------------

// EncodeDeltaInt64 encodes int64 values using delta encoding.
// Format: [first_value: int64 LE][min_delta: int64 LE][bit_width: 1 byte][packed_deltas: bit-packed]
func EncodeDeltaInt64(values []int64) []byte {
	if len(values) == 0 {
		buf := make([]byte, 17)
		// all zeros: first=0, minDelta=0, bitWidth=0
		return buf
	}

	first := values[0]

	if len(values) == 1 {
		buf := make([]byte, 17)
		binary.LittleEndian.PutUint64(buf[0:], uint64(first))
		// min_delta=0, bit_width=0
		return buf
	}

	// Compute deltas
	deltas := make([]int64, len(values)-1)
	for i := 1; i < len(values); i++ {
		deltas[i-1] = values[i] - values[i-1]
	}

	// Find min delta
	minDelta := deltas[0]
	for _, d := range deltas[1:] {
		if d < minDelta {
			minDelta = d
		}
	}

	// Compute adjusted deltas (all >= 0)
	adjusted := make([]uint64, len(deltas))
	var maxAdj uint64
	for i, d := range deltas {
		adjusted[i] = uint64(d - minDelta)
		if adjusted[i] > maxAdj {
			maxAdj = adjusted[i]
		}
	}

	// Determine bit width needed
	bitWidth := 0
	if maxAdj > 0 {
		bitWidth = 64 - countLeadingZeros64(maxAdj)
	}

	// Write header
	buf := make([]byte, 17)
	binary.LittleEndian.PutUint64(buf[0:], uint64(first))
	binary.LittleEndian.PutUint64(buf[8:], uint64(minDelta))
	buf[16] = byte(bitWidth)

	if bitWidth == 0 {
		return buf
	}

	// Bit-pack the adjusted deltas
	totalBits := len(adjusted) * bitWidth
	packedLen := (totalBits + 7) / 8
	packed := make([]byte, packedLen)

	for i, v := range adjusted {
		bitOffset := i * bitWidth
		for b := 0; b < bitWidth; b++ {
			if v&(1<<uint(b)) != 0 {
				byteIdx := (bitOffset + b) / 8
				bitIdx := (bitOffset + b) % 8
				packed[byteIdx] |= 1 << uint(bitIdx)
			}
		}
	}

	buf = append(buf, packed...)
	return buf
}

// DecodeDeltaInt64 decodes delta-encoded int64 values.
func DecodeDeltaInt64(data []byte, count int) []int64 {
	if count == 0 {
		return nil
	}

	first := int64(binary.LittleEndian.Uint64(data[0:]))
	out := make([]int64, count)
	out[0] = first

	if count == 1 {
		return out
	}

	minDelta := int64(binary.LittleEndian.Uint64(data[8:]))
	bitWidth := int(data[16])
	packed := data[17:]

	prev := first
	for i := 0; i < count-1; i++ {
		var adj uint64
		if bitWidth > 0 {
			bitOffset := i * bitWidth
			for b := 0; b < bitWidth; b++ {
				byteIdx := (bitOffset + b) / 8
				bitIdx := (bitOffset + b) % 8
				if packed[byteIdx]&(1<<uint(bitIdx)) != 0 {
					adj |= 1 << uint(b)
				}
			}
		}
		delta := int64(adj) + minDelta
		val := prev + delta
		out[i+1] = val
		prev = val
	}
	return out
}

// countLeadingZeros64 returns the number of leading zero bits in v.
func countLeadingZeros64(v uint64) int {
	if v == 0 {
		return 64
	}
	n := 0
	if v&0xFFFFFFFF00000000 == 0 {
		n += 32
		v <<= 32
	}
	if v&0xFFFF000000000000 == 0 {
		n += 16
		v <<= 16
	}
	if v&0xFF00000000000000 == 0 {
		n += 8
		v <<= 8
	}
	if v&0xF000000000000000 == 0 {
		n += 4
		v <<= 4
	}
	if v&0xC000000000000000 == 0 {
		n += 2
		v <<= 2
	}
	if v&0x8000000000000000 == 0 {
		n++
	}
	return n
}

// ---------------------------------------------------------------------------
// 5. BYTE_STREAM_SPLIT encoding
// ---------------------------------------------------------------------------

// EncodeByteStreamSplit32 encodes float32 values using byte stream splitting.
// 4 byte streams of length count. Byte j of value i is stored at stream_j[i].
func EncodeByteStreamSplit32(values []float32) []byte {
	n := len(values)
	buf := make([]byte, n*4)
	for i, v := range values {
		bits := math.Float32bits(v)
		buf[0*n+i] = byte(bits)
		buf[1*n+i] = byte(bits >> 8)
		buf[2*n+i] = byte(bits >> 16)
		buf[3*n+i] = byte(bits >> 24)
	}
	return buf
}

// DecodeByteStreamSplit32 decodes byte-stream-split encoded float32 values.
func DecodeByteStreamSplit32(data []byte, count int) []float32 {
	out := make([]float32, count)
	for i := 0; i < count; i++ {
		bits := uint32(data[0*count+i]) |
			uint32(data[1*count+i])<<8 |
			uint32(data[2*count+i])<<16 |
			uint32(data[3*count+i])<<24
		out[i] = math.Float32frombits(bits)
	}
	return out
}

// EncodeByteStreamSplit64 encodes float64 values using byte stream splitting.
// 8 byte streams of length count. Byte j of value i is stored at stream_j[i].
func EncodeByteStreamSplit64(values []float64) []byte {
	n := len(values)
	buf := make([]byte, n*8)
	for i, v := range values {
		bits := math.Float64bits(v)
		buf[0*n+i] = byte(bits)
		buf[1*n+i] = byte(bits >> 8)
		buf[2*n+i] = byte(bits >> 16)
		buf[3*n+i] = byte(bits >> 24)
		buf[4*n+i] = byte(bits >> 32)
		buf[5*n+i] = byte(bits >> 40)
		buf[6*n+i] = byte(bits >> 48)
		buf[7*n+i] = byte(bits >> 56)
	}
	return buf
}

// DecodeByteStreamSplit64 decodes byte-stream-split encoded float64 values.
func DecodeByteStreamSplit64(data []byte, count int) []float64 {
	out := make([]float64, count)
	for i := 0; i < count; i++ {
		bits := uint64(data[0*count+i]) |
			uint64(data[1*count+i])<<8 |
			uint64(data[2*count+i])<<16 |
			uint64(data[3*count+i])<<24 |
			uint64(data[4*count+i])<<32 |
			uint64(data[5*count+i])<<40 |
			uint64(data[6*count+i])<<48 |
			uint64(data[7*count+i])<<56
		out[i] = math.Float64frombits(bits)
	}
	return out
}
