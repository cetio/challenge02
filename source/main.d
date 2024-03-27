import std.stdio;
import std.file;
import std.math;
import inteli.tmmintrin;
import inteli.emmintrin;
import inteli.avxintrin;
import inteli.avx2intrin;
import inteli.smmintrin;
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

	//ubyte[] data = cast(ubyte[])read("C:\\Users\\stake\\Downloads\\100MB.zip");
    ubyte[] data = cast(ubyte[])"delayowl29rumen84DT abcdefghijklmnopqrstuvwxyz";
	ubyte[] orig = data.dup;
    writeln(data.entropy);
    auto start = Clock.currTime;
    data.encrypt("Qat2QXQtkys6alHsddq57rgZYaBVDrUF");
    writeln(Clock.currTime - start);
    //writeln(cast(string)data);
    writeln(data.entropy);

    if (data.length < 1000)
        writeln(cast(string)data);

    data.decrypt("Qat2QXQtkys6alHsddq57rgZYaBVDrUF");

    if (data.length < 1000)
        writeln(cast(string)data);
    
    size_t diff;
    foreach (i; 0..orig.length)
    {
        if (data[i] != orig[i])
            diff++;
    }
    writeln(diff);
}

align (32) string derive(uint SEED)(string key)
{
    char[32] ret;
    foreach (i; 0..32)
    {
        foreach (j; 0..32)
        {
            ret[i] ^= ret[j] += key[j] ^ SEED;
            ret[(j + i) % 32] += ret[i] ^= key[i] ^ SEED;
        }
    }
    return ret.dup;
}

align (16) __m128i[2] mix16(ulong seed)
{
    align (16) ubyte[16] x; // enc
    align (16) ubyte[16] y; // dec
    y[] = 255;

    static foreach (ubyte j; 0..16)
    {{
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;

        ubyte k = cast(ubyte)(seed % 16);

        // Avoid duplicates as to avoid corruption when shuffling.
        // Reseeding may be faster than incrementing using `++k %= SIZE` because it has more uniform
        // distribution, but also may be slower because of that.
        while (y[k] != 255)
        {
            seed++;
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            k = cast(ubyte)(seed % 16);
        }

        x[j] = k;
        y[k] = j;
    }}

    return [*cast(__m128i*)&x, *cast(__m128i*)&y];
}

align (32) __m256i[2] mix32(ulong seed)
{
    align (32) __m128i[2] x = [mix16(seed)[0], mix16(seed ^ 0xfee8d23c)[0]];
    align (32) __m128i[2] y = [mix16(seed)[1], mix16(seed ^ 0xfee8d23c)[1]];

    return [
        *cast(__m256i*)x.ptr,
        *cast(__m256i*)y.ptr,
    ];
}

align (16) __m128i split16(ulong seed)
{
    __m128i x;
    foreach (i; 0..2)
    {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        x[i] = seed % 2 == 0 ? -1 : 0;
    }
    return x;
}

version (LDC)
{
    pragma(LDC_intrinsic, "llvm.x86.avx2.pshuf.b")
    pragma(inline)
    byte32 __builtin_ia32_pshufb256(byte32, byte32) pure @safe;

    pragma(LDC_intrinsic, "llvm.x86.avx2.pblendvb")
    pragma(inline)
    byte32 __builtin_ia32_pblendvb256(byte32, byte32, byte32) pure @safe;
}

pragma(inline)
__m256i _mm256_shuffle_epi8(__m256i a, __m256i b) pure @trusted
{
    version (GDC)
    {
        import gcc.builtins;

        return cast(__m256i)__builtin_ia32_pshufb256(cast(byte32)a, cast(byte32)b);
    }
    else version (LDC)
    {
        version (X86_64)
        {
            return cast(__m256i)__builtin_ia32_pshufb256(cast(byte32)a, cast(byte32)b);
        }
        else version (X86)
        {
            return cast(__m256i)__builtin_ia32_pshufb256(cast(byte32)a, cast(byte32)b);
        }
        else
        {
            auto c = _mm_shuffle_epi8((cast(__m128i*)&a)[0], (cast(__m128i*)&b)[0]);
            auto d = _mm_shuffle_epi8((cast(__m128i*)&a)[1], (cast(__m128i*)&b)[1]);
            align (32) __m128i[2] ret = [c, d];
            return *cast(__m256i*)&ret;
        }
    }
    else
    {
        auto c = _mm_shuffle_epi8((cast(__m128i*)&a)[0], (cast(__m128i*)&b)[0]);
        auto d = _mm_shuffle_epi8((cast(__m128i*)&a)[1], (cast(__m128i*)&b)[1]);
        align (32) __m128i[2] ret = [c, d];
        return *cast(__m256i*)&ret;
    }
}

pragma(inline)
__m256i _mm256_blendv_epi8(__m256i a, __m256i b, __m256i mask) pure @trusted
{
    version (GDC)
    {
        import gcc.builtins;

        return cast(__m256i)__builtin_ia32_pblendvb256(cast(byte32)a, cast(byte32)b, cast(byte32)mask);
    }
    else version (LDC)
    {
        version (X86_64)
        {
            return cast(__m256i)__builtin_ia32_pblendvb256(cast(byte32)a, cast(byte32)b, cast(byte32)mask);
        }
        else version (X86)
        {
            return cast(__m256i)__builtin_ia32_pblendvb256(cast(byte32)a, cast(byte32)b, cast(byte32)mask);
        }
        else
        {
            auto c = _mm_blendv_epi8((cast(__m128i*)&a)[0], (cast(__m128i*)&b)[0], (cast(__m128i*)&mask)[0]);
            auto d = _mm_blendv_epi8((cast(__m128i*)&a)[1], (cast(__m128i*)&b)[1], (cast(__m128i*)&mask)[1]);
            align (32) __m128i[2] ret = [c, d];
            return *cast(__m256i*)&ret;
        }
    }
    else
    {
        auto c = _mm_blendv_epi8((cast(__m128i*)&a)[0], (cast(__m128i*)&b)[0], (cast(__m128i*)&mask)[0]);
        auto d = _mm_blendv_epi8((cast(__m128i*)&a)[1], (cast(__m128i*)&b)[1], (cast(__m128i*)&mask)[1]);
        align (32) __m128i[2] ret = [c, d];
        return *cast(__m256i*)&ret;
    }
}

void encrypt(ref ubyte[] data, string key)
{
    assert(key.length == 32, "Key must be 256-bits!");

    __m256i R0 = _mm256_loadu_si256(cast(__m256i*)key.derive!(0x0c0b6479).ptr);
    __m256i R1 = _mm256_loadu_si256(cast(__m256i*)key.derive!(0x8ea853bc).ptr);
    __m256i R2 = _mm256_loadu_si256(cast(__m256i*)key.derive!(0x79b953f7).ptr);
    __m256i R3 = _mm256_loadu_si256(cast(__m256i*)key.derive!(0xfe778533).ptr);

    ulong S = (R0 ^ R1 ^ R2 ^ R3)[0];
    ulong R = 8;

    immutable long[16] Z = [
        R0[0], R0[1], R0[2], R0[3],
        R1[0], R1[1], R1[2], R1[3],
        R2[0], R2[1], R2[2], R2[3],
        R3[0], R3[1], R3[2], R3[3]
    ];

    __m128i x = mix16(++S)[0];
    __m256i y = mix32(++S)[0];
    __m128i s = split16(++S);

    if (data.length % 32 != 0)
            data ~= new ubyte[32 - (data.length % 32)];

    void turn()
    {
        S = S ^ Z[(S % 16)];
        R = (R ^ S) % 8;
        s = split16(++S);
    }

    void mix(T)(ref T v)
    {
        while (R-- <= 0)
        {
            static if (T.sizeof == 16)
                _mm_shuffle_epi8(v, x);
            else
                _mm256_shuffle_epi8(v, y);
        }
        turn();
    }
    
    __m128i[] fold = cast(__m128i[])data;
    __m256i[] pair = cast(__m256i[])data;

    foreach (i, ref v; fold[$/2..$])
    {
        auto g = _mm_blendv_epi8(v, fold[i], s);
        fold[i] = _mm_blendv_epi8(fold[i], v, s);
        v = g;
        mix(v);
    }

    foreach (ref v; pair)
    {
        v -= R2;
        v ^= R1;
        mix(v);
        v += S;
        v += R0;
        v ^= R3;
        mix(v);
    }
}

void decrypt(ref ubyte[] data, string key)
{
    assert(key.length == 32, "Key must be 256-bits!");

    auto R0 = _mm256_loadu_si256(cast(__m256i*)key.derive!(0x0c0b6479).ptr);
    auto R1 = _mm256_loadu_si256(cast(__m256i*)key.derive!(0x8ea853bc).ptr);
    auto R2 = _mm256_loadu_si256(cast(__m256i*)key.derive!(0x79b953f7).ptr);
    auto R3 = _mm256_loadu_si256(cast(__m256i*)key.derive!(0xfe778533).ptr);
    ulong S = (R0 ^ R1 ^ R2 ^ R3)[0];
    ulong T = S;

    immutable long[16] Z = [
        R0[0],
        R0[1],
        R0[2],
        R0[3],
        R1[0],
        R1[1],
        R1[2],
        R1[3],
        R2[0],
        R2[1],
        R2[2],
        R2[3],
        R3[0],
        R3[1],
        R3[2],
        R3[3]
    ];

    __m128i x = mix16(S ^ R0[1])[1];
    __m128i f = mix16(S ^ R1[2])[1];
    __m256i y = mix32(S ^ R2[3])[1];
    __m256i c = mix32(S ^ R3[0])[1];
    __m128i s = split16(S);

    void turn()
    {
        S = S ^ Z[(S % 16)];

        x = mix16(S ^ R0[1])[1];
        f = mix16(S ^ R1[2])[1];
        y = mix32(S ^ R2[3])[1];
        c = mix32(S ^ R3[0])[1];
        s = split16(S);
    }

    __m128i[] fold = cast(__m128i[])data;
    __m256i[] pair = cast(__m256i[])data;

    foreach (i, ref v; fold[$/2..$])
        turn();

    foreach (ref v; pair)
    {
        v = _mm256_shuffle_epi8(v, y);
        v ^= R3;
        v -= R0;
        v -= S;
        v = _mm256_shuffle_epi8(v, c);
        v ^= R1;
        v += R2;
        turn();
    }

    S = T;

    foreach (i, ref v; fold[$/2..$])
    {
        v = _mm_shuffle_epi8(v, x);
        auto g = _mm_blendv_epi8(v, fold[i], s);
        fold[i] = _mm_blendv_epi8(fold[i], v, s);
        v = g;
        v = _mm_shuffle_epi8(v, f);
        turn();
    }
}