import std.stdio;
import std.file;
import std.math;
import inteli.tmmintrin;
import inteli.emmintrin;
import inteli.avxintrin;
import inteli.avx2intrin;
import inteli.math;

double entropy(ubyte[] data) 
{
    ulong totalCount = data.length;
    ulong[256] frequencies;
    double entropy = 0.0;

    // Count frequencies of each character
    foreach (b; data)
        frequencies[b]++;

    // Calculate entropy
    foreach (count; frequencies) 
    {
        if (count != 0) 
        {
            double probability = cast(double)count / totalCount;
            entropy -= probability * log2(probability);
        }
    }

    return entropy;
}

void main()
{
    import std.datetime;

	ubyte[] data = cast(ubyte[])read("C:\\Users\\stake\\Downloads\\100MB.bin");
    //ubyte[] data = cast(ubyte[])"delayowl29rumen84DT";
	ubyte[] orig = data.dup;
    writeln(data.entropy);
    auto start = Clock.currTime;
    data.encrypt("Qat2QXQtkys6alHsddq57rgZYaBVDrUf");
    writeln(Clock.currTime - start);
    //writeln(cast(string)data);
    writeln(data.entropy);
    data.decrypt("Qat2QXQtkys6alHsddq57rgZYaBVDrUf");
    //writeln(cast(string)data);
    size_t diff;
    foreach (i; 0..orig.length)
    {
        if (data[i] != orig[i])
            diff++;
    }
    writeln(diff);
}

string derive(string key)
{
    char[32] ret;
    foreach (i; 0..32)
    {
        foreach (j; 0..32)
        {
            ret[i] ^= ret[j] += key[j];
            ret[(j + i) % 32] += ret[i] ^= key[i];
        }
    }
    return ret.dup;
}

void encrypt(ref ubyte[] data, string key)
{
    assert(key.length == 32, "Key must be 256-bits!");
    key = key.derive;

    ulong a = (cast(ulong*)key.ptr)[0];
    ulong b = (cast(ulong*)key.ptr)[1];
    ulong c = (cast(ulong*)key.ptr)[3];
    ulong e = (a + b + (cast(ulong*)key.ptr)[2]) ^ c;

    if (data.length % 32 != 0)
        data ~= new ubyte[32 - (data.length % 32)];

    auto blocks = cast(__m256i[])data;
    auto cards = cast(__m128i[])data;
    ubyte[16] fwmask;
    ubyte[16] bwmask;
    bwmask[] = 255;
    foreach (ubyte j; 0..16)
    {
        ubyte index = cast(ubyte)(((e % 33_333) / (cast(ulong)j + 1)) % 16);
        while (bwmask[index] != 255)
            index = (index + 1) % 16;
        fwmask[j] = index;
        bwmask[index] = j;
    }

    pragma(inline)
    void swap(ulong* a, ulong* b)
    {
        *a ^= *b;
        *b ^= *a;
        *a ^= *b;
        *a = (*a << 1) | (*a >> 63);
        *b = (*b >> 1) | (*b << 63);
    }

    foreach (i, ref block; blocks[0..$/2])
    {
        block = _mm256_xor_si256(block, cast(__m256i)a);
        block += cast(__m256i)b;
        swap(cast(ulong*)&block, (cast(ulong*)&block) + 3);
        block = _mm256_xor_si256(block, *cast(__m256i*)key);
        swap((cast(ulong*)&block) + 1, (cast(ulong*)&block) + 2);
        block = _mm256_xor_si256(block, cast(__m256i)c);
    }

    foreach (i; 0..(cards.length / 2))
    {
        swap((cast(ulong*)&cards[i]) + 1, (cast(ulong*)&cards[$-1 - i]) + 1);
        cards[i] = _mm_shuffle_epi8(cards[i], *cast(__m128i*)&fwmask);
    }

    foreach_reverse (i, ref block; blocks[$/2..$])
    {
        block = _mm256_xor_si256(block, cast(__m256i)a);
        block -= cast(__m256i)b;
        swap(cast(ulong*)&block, (cast(ulong*)&block) + 3);
        block = _mm256_xor_si256(block, *cast(__m256i*)key);
        swap((cast(ulong*)&block) + 1, (cast(ulong*)&block) + 2);
        block = _mm256_xor_si256(block, cast(__m256i)c);
    }

    foreach (i; (cards.length / 2)..cards.length)
    {
        swap(cast(ulong*)&cards[i], cast(ulong*)&cards[$-1 - i]);
        swap((cast(ulong*)&cards[i]) + 1, (cast(ulong*)&cards[$-1 - i]) + 1);
        cards[i] = _mm_shuffle_epi8(cards[i], *cast(__m128i*)&fwmask);
    }
}

void decrypt(ref ubyte[] data, string key)
{
    assert(key.length == 32, "Key must be 256-bits!");
    assert(data.length % 32 == 0, "Data size must be a multiple of 32!");
    key = key.derive;

    ulong a = (cast(ulong*)key.ptr)[0];
    ulong b = (cast(ulong*)key.ptr)[1];
    ulong c = (cast(ulong*)key.ptr)[3];
    ulong e = (a + b + (cast(ulong*)key.ptr)[2]) ^ c;

    auto blocks = cast(__m256i[])data;
    auto cards = cast(__m128i[])data;
    ubyte[16] fwmask;
    ubyte[16] bwmask;
    bwmask[] = 255;
    foreach (ubyte j; 0..16)
    {
        ubyte index = cast(ubyte)(((e % 33_333) / (cast(ulong)j + 1)) % 16);
        while (bwmask[index] != 255)
            index = (index + 1) % 16;
        fwmask[j] = index;
        bwmask[index] = j;
    }

    pragma(inline)
    void swap(ulong* a, ulong* b)
    {
        *a ^= *b;
        *b ^= *a;
        *a ^= *b;
        *a = (*a << 1) | (*a >> 63);
        *b = (*b >> 1) | (*b << 63);
    }

    foreach_reverse (i; (cards.length / 2)..cards.length)
    {
        cards[i] = _mm_shuffle_epi8(cards[i], *cast(__m128i*)&bwmask);
        swap(cast(ulong*)&cards[i], cast(ulong*)&cards[$-1 - i]);
        swap((cast(ulong*)&cards[i]) + 1, (cast(ulong*)&cards[$-1 - i]) + 1);
    }

    foreach (i, ref block; blocks[$/2..$])
    {
        block = _mm256_xor_si256(block, cast(__m256i)c);
        swap((cast(ulong*)&block) + 1, (cast(ulong*)&block) + 2);
        block = _mm256_xor_si256(block, *cast(__m256i*)key);
        swap(cast(ulong*)&block, (cast(ulong*)&block) + 3);
        block += cast(__m256i)b;
        block = _mm256_xor_si256(block, cast(__m256i)a);
    }

    foreach_reverse (i; 0..(cards.length / 2))
    {
        cards[i] = _mm_shuffle_epi8(cards[i], *cast(__m128i*)&bwmask);
        swap((cast(ulong*)&cards[i]) + 1, (cast(ulong*)&cards[$-1 - i]) + 1);
    }

    foreach (i, ref block; blocks[0..$/2])
    {
        block = _mm256_xor_si256(block, cast(__m256i)c);
        swap((cast(ulong*)&block) + 1, (cast(ulong*)&block) + 2);
        block = _mm256_xor_si256(block, *cast(__m256i*)key);
        swap(cast(ulong*)&block, (cast(ulong*)&block) + 3);
        block -= cast(__m256i)b;
        block = _mm256_xor_si256(block, cast(__m256i)a);
    }
}