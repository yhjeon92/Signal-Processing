using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.IO;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Factorization;

namespace TestBranch
{
  public class FilterMethods
  {
        private static readonly double[] db2LP = new double[4] { -0.1294095226, 0.2241438680, 0.8365163037, 0.4829629131 }; // daubechies 2 wavelet LP coeff
        private static readonly double[] db2HP = new double[4] { -0.4829629131, 0.8365163037, -0.2241438680, -0.1294095226 };
        private static readonly double[] haarLP = new double[2] { 0.7071067812, 0.7071067812 };
        private static readonly double[] haarHP = new double[2] { -0.7071067812, 0.7071067812 };

        private static readonly double[] b_bandpass = new double[5] { 0.0267071, 0, -0.0534141, 0, 0.0267071 };          // butterworth, 2nd order, bandpass, 10 ~ 40 Hz (Fs:512 Hz)
        private static readonly double[] a_bandpass = new double[5] { 1, -3.3817424, 4.3851928, -2.5948997, 0.5942807 };    // reverse coeffs.

        private static readonly double[] b_highpass = new double[3] { 0.9974001, -1.9948003, 0.9974001 };                // butterworth, 2nd order, highpass 0.3 Hz (Fs:512)
        private static readonly double[] a_highpass = new double[3] { 1, -1.9947935, 0.9948070 };                           // reverse coeffs.
        
        public enum FilterType
        {
            lowpass = 0,
            highpass = 1,
            bandpass = 2,
            bandstop = 3
        }
        /// <summary>
        /// Get Chebyshev II bandstop filter coefficients. Ref: github.com/scipy/scipy/blob/v0.17.1/scipy/signal/filter_design.py
        /// </summary>
        /// <param name="order">Order of filter.</param>
        /// <param name="attenuation">Attenuation of filter.</param>
        /// <param name="cutOffFrequency"></param>
        /// <param name="samplingRate">Sampling Rate.</param>
        /// <param name="filter">Type of filter. (*Currently has no impact on the result)</param>
        /// <returns></returns>
        public static Tuple<double[], double[]> ChebyshevTwoBandstopCoefficients(int order, double attenuation, double[] cutOffFrequency, double samplingRate, FilterType filter)
        {
            // only notch implemented
            // chebyshev II type filter implementation
            // reference: github.com/scipy/scipy/blob/v0.17.1/scipy/signal/filter_design.py

            int N = (int)Math.Floor((double)order / 2) * 2;

            if (N < 2)
                return new Tuple<double[], double[]>(new double[1] { 0.0d }, new double[1] { 1.0d });

            var zeroPoles = cheb2ap(N, attenuation);
            Complex[] z = zeroPoles.Item1;
            Complex[] p = zeroPoles.Item2;
            double k = zeroPoles.Item3;

            double fs = 2.0d;
            double[] warped = new double[2];
            warped[0] = 2 * fs * Math.Tan(Math.PI * cutOffFrequency[0] / samplingRate);
            warped[1] = 2 * fs * Math.Tan(Math.PI * cutOffFrequency[1] / samplingRate);
            double bandWidth = warped[1] - warped[0];
            double w0 = Math.Sqrt(warped[0] * warped[1]);

            if (z.Length > p.Length)
            {
                throw new ArgumentException("Must have at least as many poles as zeros");
            }

            zeroPoles = zpk_lowpass2bandstop(z, p, k, w0, bandWidth);
            z = zeroPoles.Item1;
            p = zeroPoles.Item2;
            k = zeroPoles.Item3;

            zeroPoles = zpkBilinear(z, p, k, fs);

            z = zeroPoles.Item1;
            p = zeroPoles.Item2;
            k = zeroPoles.Item3;

            return zpk2tf(z, p, k);
        }

        private static Tuple<Complex[], Complex[], double> buttap(int order)
        {
            int N = (int)Math.Floor((double)order / 2) * 2;
            if (N < 2)
            {
                return new Tuple<Complex[], Complex[], double>(new Complex[0] { }, new Complex[0] { }, 0.0d);
            }

            int[] m = new int[N];
            Complex[] z = new Complex[0] { };
            Complex[] p = new Complex[N];
            for (int i = 0; i < N; i++)
            {
                m[i] = -N + 1 + (2 * i);
                p[i] = -Complex.Exp(Complex.ImaginaryOne * Math.PI * m[i] / (2 * N));
            }
            double k = 1.0d;

            return new Tuple<Complex[], Complex[], double>(z, p, k);
        }

        private static Tuple<Complex[], Complex[], double> besselap(int order)
        {
            int N = (int)Math.Floor((double)order / 2) * 2;
            if (N < 2 || N > 4)
            {
                return new Tuple<Complex[], Complex[], double>(new Complex[0] { }, new Complex[0] { }, 0.0d);
            }

            Complex[] z = new Complex[0] { };
            Complex[] p = new Complex[N];
            double k = 1.0d;
            if (N == 2)
            {
                p[0] = new Complex(-0.8660254037844386467637229, 0.4999999999999999999999996);
                p[1] = new Complex(-0.8660254037844386467637229, -0.4999999999999999999999996);
            }
            else if (N == 4)
            {
                p[0] = new Complex(-0.6572111716718829545787781, -0.8301614350048733772399715);
                p[1] = new Complex(-0.6572111716718829545787788, 0.8301614350048733772399715);
                p[2] = new Complex(-0.9047587967882449459642637, -0.2709187330038746636700923);
                p[3] = new Complex(-0.9047587967882449459642624 , 0.2709187330038746636700926);
            }
            return new Tuple<Complex[], Complex[], double>(z, p, k);
        }

        private static Tuple<Complex[], Complex[], double> cheb2ap(int order, double rs)
        {
            int N = (int)Math.Floor((double)order / 2) * 2;
            if (N < 2)
            {
                return new Tuple<Complex[], Complex[], double> (new Complex[0] { }, new Complex[0] { }, 0.0d);
            }
            double delta = 1.0 / Math.Sqrt(Math.Pow(10.0d, 0.1 * rs) - 1);
            double mu = Math.Log(1.0 / delta + Math.Sqrt((1.0 / delta) * (1.0 / delta) + 1)) / N;

            int[] m = new int[N];
            Complex[] z = new Complex[N];
            Complex[] p = new Complex[N];

            for (int i = 0; i < N; i++)
            {
                m[i] = -N + 1 + (2 * i);
                z[i] = -Complex.Conjugate(Complex.ImaginaryOne / Math.Sin(m[i] * Math.PI / (2 * N)));
                p[i] = -Complex.Exp(Complex.ImaginaryOne * Math.PI * m[i] / (2 * N));
            }

            for (int i = 0; i < N; i++)
            {
                p[i] = Math.Sinh(mu) * p[i].Real + Complex.ImaginaryOne * Math.Cosh(mu) * p[i].Imaginary;
                p[i] = 1 / p[i];
            }

            double k = (p.Aggregate((a, b) => Complex.Multiply(-a, -b)) / z.Aggregate((a, b) => Complex.Multiply(-a, -b))).Real;

            return new Tuple<Complex[], Complex[], double>(z, p, k);
        }

        private static Tuple<Complex[], Complex[], double> zpk_lowpass2bandstop(Complex[] z, Complex[] p, double k, double w0, double bw)
        {
            if (z.Length > p.Length)
            {
                throw new ArgumentException("Must have at least as many poles as zeros");
            }

            int degree = p.Length - z.Length;

            Complex[] z_hp = new Complex[z.Length];
            Complex[] p_hp = new Complex[p.Length];

            for (int i = 0; i < z_hp.Length; i++)
            {
                z_hp[i] = (bw / 2) / z[i];
            }
            for (int i = 0; i < p_hp.Length; i++)
            {
                p_hp[i] = (bw / 2) / p[i];
            }
            Complex[] z_bs = z_hp.Select(s => s + Complex.Sqrt(s * s - w0 * w0)).ToArray().Concat(z_hp.Select(s => s - Complex.Sqrt(s * s - w0 * w0))).ToArray();
            Complex[] p_bs = p_hp.Select(s => s + Complex.Sqrt(s * s - w0 * w0)).ToArray().Concat(p_hp.Select(s => s - Complex.Sqrt(s * s - w0 * w0))).ToArray();

            for (int i = 0; i < z_bs.Length; i++)
            {
                if (z_bs[i].Real < 1E-16)
                {
                    z_bs[i] = new Complex(0.0, z_bs[i].Imaginary);
                }
            }

            z_bs = z_bs.Concat(Enumerable.Repeat(Complex.ImaginaryOne * w0, degree)).ToArray();
            z_bs = z_bs.Concat(Enumerable.Repeat(-1 * Complex.ImaginaryOne * w0, degree)).ToArray();
            double k_bs = k * (z.Aggregate((a, b) => Complex.Multiply(-a, -b)) / p.Aggregate((a, b) => Complex.Multiply(-a, -b))).Real;

            return new Tuple<Complex[], Complex[], double>(z_bs, p_bs, k_bs);
        }

        private static Tuple<Complex[], Complex[], double> zpk_lowpass2bandpass(Complex[] z, Complex[] p, double k, double w0, double bw)
        {
            if (z.Length > p.Length)
            {
                throw new ArgumentException("Must have at least as many poles as zeros");
            }

            int degree = p.Length - z.Length;

            Complex[] z_lp = new Complex[z.Length];
            Complex[] p_lp = new Complex[p.Length];

            for (int i = 0; i < z_lp.Length; i++)
            {
                z_lp[i] = z[i] * (bw / 2);
            }
            for (int i = 0; i < p_lp.Length; i++)
            {
                p_lp[i] = p[i] * (bw / 2);
            }
            Complex[] z_bp = z_lp.Select(s => s + Complex.Sqrt(s * s - w0 * w0)).ToArray().Concat(z_lp.Select(s => s - Complex.Sqrt(s * s - w0 * w0))).ToArray();
            Complex[] p_bp = p_lp.Select(s => s + Complex.Sqrt(s * s - w0 * w0)).ToArray().Concat(p_lp.Select(s => s - Complex.Sqrt(s * s - w0 * w0))).ToArray();

            z_bp = z_bp.Concat(Enumerable.Repeat(Complex.Zero, degree)).ToArray();

            double k_bp = k * Math.Pow(bw, degree);

            return new Tuple<Complex[], Complex[], double>(z_bp, p_bp, k_bp);
        }

        private static Tuple<Complex[], Complex[], double> zpkBilinear(Complex[] z, Complex[] p, double k, double fs)
        {
            if (z.Length > p.Length)
            {
                throw new ArgumentException("Must have at least as many poles as zeros");
            }

            int degree = p.Length - z.Length;
            double fs2 = 2 * fs;

            Complex[] z_z = z.Select(s => (fs2 + s) / (fs2 - s)).ToArray();
            Complex[] p_z = p.Select(s => (fs2 + s) / (fs2 - s)).ToArray();

            z_z = z_z.Concat(Enumerable.Repeat(new Complex(-1.0, 0.0), degree)).ToArray();

            double k_z = k * (z.Aggregate((a, b) => Complex.Multiply(fs2 - a, fs2 - b)) / p.Aggregate((a, b) => Complex.Multiply(fs2 - a, fs2 - b))).Real;
            //double k_z = k * (z_z.Aggregate((a, b) => Complex.Multiply(fs2 - a, fs2 - b)) / p_z.Aggregate((a, b) => Complex.Multiply(fs2 - a, fs2 - b))).Real;

            return new Tuple<Complex[], Complex[], double>(z_z, p_z, k_z);
        }

        private static Tuple<double[], double[]> zpk2tf(Complex[] z, Complex[] p, double k)
        {
            double[] numCoeffResult; double[] denCoeffResult;
            Complex[] numCoeffs = new Complex[1] { new Complex(1.0, 0.0) };
            Complex[] denCoeffs = new Complex[1] { new Complex(1.0, 0.0) };

            foreach (Complex zero in z)
            {
                numCoeffs = MultiplyPolynomial(numCoeffs, new Complex[2] { Complex.One, -zero });
            }

            numCoeffs = numCoeffs.Select(s => s * k).ToArray();

            foreach (Complex pole in p)
            {
                denCoeffs = MultiplyPolynomial(denCoeffs, new Complex[2] { Complex.One, -pole });
            }

            numCoeffResult = numCoeffs.Select(s => s.Real).ToArray();
            denCoeffResult = denCoeffs.Select(s => s.Real).ToArray();

            return new Tuple<double[], double[]>(numCoeffResult, denCoeffResult);

        }
        /// <summary>
        /// Get Butterworth filter coefficients, Highpass and Lowpass available.
        /// </summary>
        /// <param name="order">Order of filter, shall be multiple of 2.</param>
        /// <param name="cutOffFrequency">Cutoff frequency (in Hz).</param>
        /// <param name="samplingRate">Number of samples per second, in Hz.</param>
        /// <param name="filter">Type of filter. Only Highpass and Lowpass available now.</param>
        /// <returns></returns>
        public static Tuple<double[], double[]> ButterworthCoefficients(int order, double cutOffFrequency, double samplingRate, FilterType filter)
        {
            int n = (int)Math.Floor((double)order / 2) * 2;     // order of filter, multiple of 2 (2, 4, 6, ...)
            if (n < 2)
            {
                return new Tuple<double[], double[]>(new double[1] { 0.0d }, new double[1] { 1.0d });
            }
            double[] numCoeffResult = new double[1] { 1.0d };
            double[] denCoeffResult = new double[1] { 1.0d };

            double Oc = Math.Tan(Math.PI * cutOffFrequency / samplingRate);
            
            List<Tuple<double[], double[]>> secondOrderFilterList = new List<Tuple<double[], double[]>>();

            if (filter.Equals(FilterType.lowpass))          // Calculate 2nd order butterworth lowpass filter coefficients
            {
                for (int i = 0; i < n / 2; i++)
                {
                    double[] numCoeffOneFilter = new double[3];
                    double[] denomCoeffOneFilter = new double[3];

                    double c = 1 + 2 * Math.Sin(Math.PI * (2 * i + 1) / (2 * n)) * Oc + Oc * Oc;

                    numCoeffOneFilter[0] = Oc * Oc / c;
                    numCoeffOneFilter[1] = 2 * Oc * Oc / c;
                    numCoeffOneFilter[2] = Oc * Oc / c;

                    denomCoeffOneFilter[0] = 1;
                    denomCoeffOneFilter[1] = 2 * (Oc * Oc - 1) / c;
                    denomCoeffOneFilter[2] = (1 - 2 * Math.Sin(Math.PI * (2 * i + 1) / (2 * n)) * Oc + Oc * Oc) / c;
                    secondOrderFilterList.Add(new Tuple<double[], double[]>(numCoeffOneFilter, denomCoeffOneFilter));
                }
            }
            else if (filter.Equals(FilterType.highpass))    // Calculate 2nd order butterworth highpass filter coefficients
            {
                for (int i = 0; i < n / 2; i++)
                {
                    double[] numCoeffOneFilter = new double[3];
                    double[] denomCoeffOneFilter = new double[3];

                    double c = 1 + 2 * Math.Sin(Math.PI * (2 * i + 1) / (2 * n)) * Oc + Oc * Oc;
                    numCoeffOneFilter[0] = 1 / c;
                    numCoeffOneFilter[1] = - 2 / c;
                    numCoeffOneFilter[2] = 1 / c;
                    denomCoeffOneFilter[0] = 1;
                    denomCoeffOneFilter[1] = 2 * (Oc * Oc - 1) / c;
                    denomCoeffOneFilter[2] = (1 - 2 * Math.Sin(Math.PI * (2 * i + 1) / (2 * n)) * Oc + Oc * Oc) / c;
                    secondOrderFilterList.Add(new Tuple<double[], double[]>(numCoeffOneFilter, denomCoeffOneFilter));
                }
            }
            else
            {
                return new Tuple<double[], double[]>(new double[1] { 0.0d }, new double[1] { 1.0d });
            }

            foreach (Tuple<double[], double[]> eachFilter in secondOrderFilterList)
            {
                numCoeffResult = MultiplyPolynomial(numCoeffResult, eachFilter.Item1);
                denCoeffResult = MultiplyPolynomial(denCoeffResult, eachFilter.Item2);
            }

            return new Tuple<double[], double[]>(numCoeffResult, denCoeffResult);
        }

        private static double[] MultiplyPolynomial(double[] polyOne, double[] polyTwo)
        {
            double[] polyResult = new double[polyOne.Length + polyTwo.Length - 1];
            for (int i = 0; i < polyTwo.Length; i++)
            {
                for (int j = 0; j < polyOne.Length; j++)
                {
                    polyResult[j + i] += polyOne[j] * polyTwo[i];
                }
            }
            return polyResult;
        }

        private static Complex[] MultiplyPolynomial(Complex[] polyOne, Complex[] polyTwo)
        {
            Complex[] polyResult = new Complex[polyOne.Length + polyTwo.Length - 1];
            for (int i = 0; i < polyTwo.Length; i++)
            {
                for (int j = 0; j < polyOne.Length; j++)
                {
                    polyResult[j + i] += Complex.Multiply(polyOne[j], polyTwo[i]);
                }
            }
            return polyResult;
        }
        
        private static Tuple<double[,], double[,]> Stationary1DWT(double[] data, int level, double[] waveletLP, double[] waveletHP)
        {
            int Nraw = data.Length;
            int N;
            int tap = waveletLP.Length;
            
            if (!((data.Length % Convert.ToInt32(Math.Pow(2,level))).Equals(0)))
            {
                data = data.Concat(Enumerable.Repeat((double)0, Convert.ToInt32(Math.Ceiling(data.Length / (Math.Pow(2, level))) * Math.Pow(2, level) - data.Length))).ToArray();
                N = data.Length;
            }
            else
            {
                N = data.Length;
            }
            //Matrix<double> C = Matrix<double>.Build.Dense(level, N, (double)0);
            //Matrix<double> L = Matrix<double>.Build.Dense(level, N, (double)0);

            double[,] C = new double[level, N];
            double[,] L = new double[level, N];

            double[] tempDataArray = new double[N];
            
            //Vector<double> array1 = Vector<double>.Build.Dense(N, (double)0);   // array to store approximated coeff. of this level.
            //Vector<double> array2 = Vector<double>.Build.Dense(N, (double)0);   // array to store detailed coeff. of this level.
            int gap;
            for (int i = 1; i < level + 1; i++)
            {
                gap = (int)Math.Pow(2, i - 1);
                if (i == 1)
                {
                    tempDataArray = data;
                }
                else
                {
                    tempDataArray = C.Row(i - 2);
                }

                for (int j = 0; j < N; j++)
                {
                    int[] indices = GetIndices(j, gap, tap, N, true);
                    double temp1 = 0; double temp2 = 0;
                    for (int l = 0; l < tap; l++)
                    {
                        temp1 += tempDataArray[indices[tap - 1 - l]] * waveletLP[l];
                        temp2 += tempDataArray[indices[tap - 1 - l]] * waveletHP[l];
                    }
                    C[i - 1, j] = temp1;
                    L[i - 1, j] = temp2;
                }
            }

            return new Tuple<double[,], double[,]>(C, L);
        }
        
        public static double[,] SWDBaselineRemoval(int level, double[,] dataInput)
        {
            int Ch = dataInput.GetLength(0);
            int N = dataInput.GetLength(1);

            int margin = 4 * (int)Math.Pow(2, level);
            int paddedLength = Convert.ToInt32(Math.Ceiling(N / Math.Pow(2, level)) * Math.Pow(2, level));

            double[,] dataIn = new double[Ch, paddedLength + 2 * margin];

            for (int i = 0; i < dataInput.GetLength(0); i++)
            {
                for (int k = 0; k < margin; k++)
                {
                    dataIn[i, margin - k - 1] = dataInput[i, 0] * (1 - Math.Pow((double)k / (double)margin, 2));
                    dataIn[i, N + margin + k] = dataInput[i, N - 1] * (1 - Math.Pow((double)k / (double)margin, 2));
                }
                for (int j = 0; j < dataInput.GetLength(1); j++)
                {
                    dataIn[i, j + margin] = dataInput[i, j];
                }
            }
            
            if (N < 1)
                level = 0;
            else
                level = Math.Max(0, Math.Min(level, Convert.ToInt32(Math.Floor(Math.Log(N, 2)))));

            Parallel.For(0, dataIn.GetLength(0), s => dataIn.SetRow(s, Stationary1D_Filtering(dataIn.Row(s), level, db2LP, db2HP)));

            return dataIn.SubArray(0, Ch, margin, N);
        }

        private static double[] Stationary1DWT_Inverse(double[,] C, double[,] L, double[] waveletLP, double[] waveletHP)
        {
            int level = L.GetLength(0);
            int N = L.GetLength(1);
            int gap;
            int tap = waveletLP.Length;
            int[] indices;

            double temp1;
            double temp2;

            double[,] reconstructed = new double[level, N];
            double[] tempDataArray;
            //Matrix<double> reconstructed = Matrix<double>.Build.Dense(level, N);
            //Vector<double> tempDataArray = Vector<double>.Build.Dense(N);
            for (int i = level; i > 0; i--)
            {
                gap = (int)Math.Pow(2, i - 1);
                if (i == level)
                    tempDataArray = C.Row(level - 1);
                else
                    tempDataArray = reconstructed.Row(i);
                for (int j = 0; j < N; j++)
                {
                    indices = GetIndices(j, gap, tap, N, false);
                    temp1 = 0;
                    temp2 = 0;
                    for (int l = 0; l < tap; l++)
                    {
                        temp1 += tempDataArray[indices[l]] * waveletLP[l];
                        temp2 += L[i-1,indices[l]] * waveletHP[l];
                    }
                    reconstructed[i - 1, j] = 0.5 * (temp1 + temp2);
                }
            }
            return reconstructed.Row(0);
        }
        /// <summary>
        /// Calculate moving averaged for a given input, based on user-defined weight.
        /// </summary>
        /// <param name="dataArray">1D double array, input.</param>
        /// <param name="weight">1D double array, moving-average coefficients, user-defined.</param>
        /// <returns></returns>
        public static double[] MovingAverage(double[] dataArray, double[] weight)
        {
            //int length = weight.Length;
            int paddingLength = Convert.ToInt16(Math.Floor((double)(weight.Length - 1) / 2));
            Vector<double> dataVector = Vector<double>.Build.DenseOfArray(dataArray);

            dataVector = Vector<double>.Build.DenseOfEnumerable(Vector<double>.Build.DenseOfEnumerable(Enumerable.Repeat(dataVector[0], paddingLength)).Concat(dataVector).Concat(Vector<double>.Build.DenseOfEnumerable(Enumerable.Repeat(dataVector.Last(), paddingLength ))));
            dataVector = ApplyIIR(weight, new double[1] { 1.0 }, dataVector) / weight.Sum();
            dataVector = dataVector.SubVector(2 * paddingLength, dataVector.Count - 2 * paddingLength);

            return dataVector.ToArray();
        }

        private static double[] Scale1D(double[] input)
        {
            double scale = (0.5) * (input.Max() - input.Min());
            double offset = input.Min() + scale;

            if (scale <= 0)
                return Enumerable.Repeat(0.0d, input.Length).ToArray();
            else
                return input.Select(s => (s - offset) / scale).ToArray();
        }

        private static Vector<double> Scale1D(Vector<double> input)
        {
            double scale = (0.5) * (input.Max() - input.Min());
            double offset = input.Min() + scale;

            if (scale <= 0)
                return Vector<double>.Build.DenseOfEnumerable(Enumerable.Repeat(0.0d, input.Count));
            else
                return Vector<double>.Build.DenseOfEnumerable(input.Select(s => (s - offset) / scale));
        }

        private static int[] GetIndices(int i, int gap, int tap, int arrayLength, bool isDecomposition)
        {
            int[] indices = new int[tap];
            int factor = 1;

            if (!isDecomposition)
                factor = 0;

            for (int k = 0; k < tap; k++)
            {
                indices[k] = i + (k - tap / 2 + factor) * gap;
                if (indices[k] >= arrayLength)
                {
                    indices[k] = (int)(indices[k] % arrayLength);
                }
                else if (indices[k] < 0)
                {
                    indices[k] = ((indices[k] + 1) % arrayLength) + arrayLength - 1;
                }
            }
            return indices;
        }

        private static double[] Stationary1D_Filtering(double[] data, int level, double[] LP, double[] HP)
        {
            var result = Stationary1DWT(data, level, LP, HP);
            return Stationary1DWT_Inverse(new double[result.Item1.GetLength(0), result.Item1.GetLength(1)], result.Item2, LP, HP);
        }
        
        private static int[] GetHistogram(double[] dataArray, int numDataBin)
        {
            if (numDataBin < 1 || dataArray.Count() < 3)
            {
                return new int[0] { };
            }
            int[] histogram = new int[numDataBin];
            int index;
            double increment = (dataArray.Max() - dataArray.Min()) / numDataBin;
            double dataMin = dataArray.Min();
            if (increment == 0)
            {
                return new int[0] { };
            }
            foreach (double data in dataArray)
            {
                index = Convert.ToInt32(Math.Floor((data - dataMin) / increment));
                if (index < 0)
                    index = 0;
                else if (index > (numDataBin - 1))
                    index = numDataBin - 1;
                histogram[index] += 1;
            }
            return histogram;
        }
        
        /// <summary>
        /// Apply filter to raw signal, requires numerator coefficients and denominator coefficients as input.
        /// </summary>
        /// <param name="numCoeff">Numerator, feed-forward coefficients of fliter.</param>
        /// <param name="denomCoeff">Denominator, feed-backward coefficients of filter.</param>
        /// <param name="signal">2D double array, raw signal to be filtered.</param>
        /// <returns></returns>
        public static double[,] ApplyFilterRawData(double[] numCoeff, double[] denomCoeff, double[,] signal)
        {
            double[,] filteredSignal = new double[signal.GetLength(0), signal.GetLength(1)];
            Action<int> filterOneRow = delegate (int s)
            {
                double lastElem = signal.Row(s).Last();
                double[] oneRowFiltered = ApplyIIR(numCoeff, denomCoeff, signal.Row(s).Select(elem => elem - lastElem).ToArray()).Select(elem => elem + lastElem).ToArray();
                lastElem = oneRowFiltered.First();
                oneRowFiltered = ApplyIIR(numCoeff, denomCoeff, oneRowFiltered.Select(elem => elem - lastElem).ToArray(), true).Select(elem => elem + lastElem).ToArray();
                filteredSignal.SetRow(s, oneRowFiltered);
            };
            Parallel.For(0, filteredSignal.GetLength(0), filterOneRow);
            return filteredSignal;
        }
        
        /// <summary>
        /// Infinite impulse-response filtering.
        /// </summary>
        /// <param name="numCoeff"></param>
        /// <param name="denomCoeff"></param>
        /// <param name="signal"></param>
        /// <param name="reverse"></param>
        /// <returns></returns>
        private static double[,] ApplyIIR(double[] numCoeff, double[] denomCoeff, double[,] signal, bool reverse)
        {
            double[,] filteredSignal = new double[signal.GetLength(0), signal.GetLength(1)];
            Parallel.For(0, filteredSignal.GetLength(0), s => filteredSignal.SetRow(s, ApplyIIR(numCoeff, denomCoeff, signal.Row(s), reverse)));
            return filteredSignal;
        }
        private static double[] ApplyIIR(double[] numCoeff, double[] denomCoeff, double[] signal)
        {
            double[] filteredSignal = new double[signal.Count()];

            double[] regX = new double[numCoeff.Count()];
            double[] regY = new double[denomCoeff.Count()];
            double centerTap;

            for (int i = 0; i < signal.Length; i++)
            {
                for (int k = regX.Count() - 1; k > 0; k--)
                    regX[k] = regX[k - 1];
                for (int k = regY.Count() - 1; k > 0; k--)
                    regY[k] = regY[k - 1];

                centerTap = 0.0d;
                regX[0] = signal[i];

                for (int k = 0; k < regX.Count(); k++)
                    centerTap += numCoeff[k] * regX[k];

                regY[0] = centerTap * denomCoeff[0];

                for (int k = 1; k < regY.Count(); k++)
                    regY[0] -= denomCoeff[k] * regY[k];

                filteredSignal[i] = regY[0];

            }
            return filteredSignal;
        }
        private static double[] ApplyIIR(double[] numCoeff, double[] denomCoeff, double[] signal, bool reverse)
        {
            double[] filteredSignal = new double[signal.Count()];

            double[] regX = new double[numCoeff.Count()];
            double[] regY = new double[denomCoeff.Count()];
            double centerTap;
            if (reverse)
            {
                for (int i = signal.Length - 1; i >= 0; i--)
                {
                    for (int k = regX.Length - 1; k > 0; k--)
                        regX[k] = regX[k - 1];

                    for (int k = regY.Length - 1; k > 0; k--)
                        regY[k] = regY[k - 1];

                    centerTap = 0.0d;
                    regX[0] = signal[i];

                    for (int k = 0; k < regX.Length; k++)
                        centerTap += numCoeff[k] * regX[k];

                    regY[0] = centerTap * denomCoeff[0];

                    for (int k = 1; k < regY.Length; k++)
                        regY[0] -= denomCoeff[k] * regY[k];

                    filteredSignal[i] = regY[0];
                }
            }
            else
            {
                for (int i = 0; i < signal.Length; i++)
                {
                    for (int k = regX.Length - 1; k > 0; k--)
                        regX[k] = regX[k - 1];

                    for (int k = regY.Length - 1; k > 0; k--)
                        regY[k] = regY[k - 1];

                    centerTap = 0.0d;
                    regX[0] = signal[i];

                    for (int k = 0; k < regX.Length; k++)
                        centerTap += numCoeff[k] * regX[k];

                    regY[0] = centerTap * denomCoeff[0];

                    for (int k = 1; k < regY.Length; k++)
                        regY[0] -= denomCoeff[k] * regY[k];

                    filteredSignal[i] = regY[0];
                }
            }
            return filteredSignal;
        }
        private static Vector<double> ApplyIIR(double[] numCoeff, double[] denomCoeff, Vector<double> signal)
        {
            Vector<double> filteredSignal = Vector<double>.Build.Dense(signal.Count);

            double[] regX = new double[numCoeff.Count()];
            double[] regY = new double[denomCoeff.Count()];
            double centerTap;

            for (int i = 0; i < signal.Count; i++)
            {
                for (int k = regX.Count() - 1; k > 0; k--)
                    regX[k] = regX[k - 1];
                for (int k = regY.Count() - 1; k > 0; k--)
                    regY[k] = regY[k - 1];

                centerTap = 0.0d;
                regX[0] = signal[i];
                for (int k = 0; k < regX.Count(); k++)
                {
                    centerTap += numCoeff[k] * regX[k];
                }
                regY[0] = centerTap * denomCoeff[0];
                for (int k = 1; k < regY.Count(); k++)
                {
                    regY[0] -= denomCoeff[k] * regY[k];
                }

                filteredSignal[i] = regY[0];
            }
            return filteredSignal;
        }
        /// <summary>
        /// 1 dimensional Fast Fourier Transformation. Input should be a power of 2 long for the best performance.
        /// </summary>
        /// <param name="dataInput">1D complex array, raw signal.</param>
        /// <returns></returns>
        private static Complex[] FFT_1D(Complex[] dataInput)
        {
            Complex[] FFTbin = new Complex[dataInput.Length];
            int N = dataInput.Length;

            Complex[] evenBin; Complex[] oddBin;
            Complex[] evenInput; Complex[] oddInput;

            if (N < 2)
                return new Complex[1] { dataInput[0] };

            evenInput = new Complex[N / 2]; oddInput = new Complex[N / 2];
            for (int i = 0; i < N / 2; i++)
            {
                evenInput[i] = dataInput[i * 2];
                oddInput[i] = dataInput[i * 2 + 1];
            }
            
            evenBin = FFT_1D(evenInput); oddBin = FFT_1D(oddInput);

            for (int i = 0; i < N / 2; i++)
            {
                FFTbin[i] = evenBin[i] + oddBin[i] * Complex.FromPolarCoordinates(1, -2 * Math.PI * i / N);
                FFTbin[i + N / 2] = evenBin[i] - oddBin[i] * Complex.FromPolarCoordinates(1, -2 * Math.PI * i / N);
            }

            return FFTbin;
        }
        /// <summary>
        /// 1 dimensional Fast Fourier Transformation. Input should be a power of 2 long for the best performance.
        /// </summary>
        /// <param name="dataInput">1D double array, raw signal.</param>
        /// <returns></returns>
        public static Complex[] FFT_1D(double[] dataInput)
        {
            int N = (int)Math.Pow(2, Math.Floor(Math.Log(dataInput.Length) / Math.Log(2)));
            Complex[] FFTbin = new Complex[N];
            dataInput = dataInput.SubArray(0, N);

            Complex[] evenBin; Complex[] oddBin;
            Complex[] evenInput; Complex[] oddInput;

            if (N < 2)
                return new Complex[1] { new Complex(dataInput[0], 0) };

            evenInput = new Complex[N / 2]; oddInput = new Complex[N / 2];
            for (int i = 0; i < N / 2; i++)
            {
                evenInput[i] = new Complex(dataInput[i * 2], 0);
                oddInput[i] = new Complex(dataInput[i * 2 + 1], 0);
            }

            evenBin = FFT_1D(evenInput); oddBin = FFT_1D(oddInput);

            for (int i = 0; i < N / 2; i++)
            {
                FFTbin[i] = evenBin[i] + oddBin[i] * Complex.FromPolarCoordinates(1, -2 * Math.PI * i / N);
                FFTbin[i + N / 2] = evenBin[i] - oddBin[i] * Complex.FromPolarCoordinates(1, -2 * Math.PI * i / N);
            }

            return FFTbin;
        }

        private static Complex[] IFFT_1D(Complex[] FFTbinInput)
        {
            int N = (int)Math.Pow(2, Math.Floor(Math.Log(FFTbinInput.Length) / Math.Log(2)));
            Complex[] IFFTbin = new Complex[N];
            FFTbinInput = FFTbinInput.SubArray(0, N);

            Complex[] evenBin; Complex[] oddBin;
            Complex[] evenInput; Complex[] oddInput;

            if (N < 2)
            {
                return new Complex[1] { FFTbinInput[0] };
            } 

            evenInput = new Complex[N / 2]; oddInput = new Complex[N / 2];
            for (int i = 0; i < N / 2; i++)
            {
                evenInput[i] = FFTbinInput[i * 2];
                oddInput[i] = FFTbinInput[i * 2 + 1];
            }

            evenBin = IFFT_1D(evenInput); oddBin = IFFT_1D(oddInput);

            for (int i = 0; i < N / 2; i++)
            {
                IFFTbin[i] = evenBin[i] + oddBin[i] * Complex.FromPolarCoordinates(1, 2 * Math.PI * i / N);
                IFFTbin[i + N / 2] = evenBin[i] - oddBin[i] * Complex.FromPolarCoordinates(1, 2 * Math.PI * i / N);
            }
            return IFFTbin;
        }
        /// <summary>
        /// MCG signal filtering method, by utilizing inverse Fourier transform.
        /// </summary>
        /// <param name="dataInput">2D double array consisted of input data. Dimension must be (Number of active channels) x (data points)</param>
        /// <param name="frequency">1D double array, consisted of frequency whose domain will be suppressed.</param>
        /// <param name="samplingRate">Sampling rate of raw signal, samples per second. (Hz)</param>
        /// <returns></returns>
        public static double[,] InverseFTFilter(double[,] dataInput, double[] frequency, double samplingRate)
        {
            Random rand = new Random();
            
            int Ch = dataInput.GetLength(0);
            int originalLength = dataInput.GetLength(1);

            int N = (int)Math.Pow(2, Math.Ceiling(Math.Log(originalLength, 2)));

            double[,] data = new double[Ch, N];
            for (int i = 0; i < Ch; i++)
                for (int j = 0; j < originalLength; j++)
                    data[i, j] = dataInput[i, j];

            Complex[] Aft = new Complex[N];
            Complex[,] Dft = new Complex[Ch, N];
            for (int i = 0; i < N; i++)
                Aft[i] = new Complex(rand.NextDouble() - 0.5, rand.NextDouble() - 0.5) / Math.Pow(10, 10);
            
            Parallel.For(0, Ch, s => Dft.SetRow(s, FFT_1D(data.Row(s))));
            
            int[] frequencyBin = new int[frequency.Length];

            for (int i = 0; i < frequency.Length; i++)
            {
                frequencyBin[i] = Convert.ToInt16(frequency[i] * N / samplingRate);

                int start = Math.Max(0, frequencyBin[i] - 30);
                int end = Math.Min(N, frequencyBin[i] + 30);
                double temp = Dft[0, frequencyBin[i]].Magnitude;
                for (int j = start; j < end; j++)
                {
                    if (Dft[0, j].Magnitude > temp)
                    {
                        temp = Dft[0, j].Magnitude;
                        frequencyBin[i] = j;
                    }
                }

            }
            
            double[,] Dfil = new double[Ch, N];
            Complex[,] Nft = new Complex[Ch, N];

            for (int i = 0; i < Ch; i++)
            {
                Nft.SetRow(i, Aft);

                foreach (int fn in frequencyBin)
                {
                    int cn = (int)Math.Ceiling(6 * fn * (double)(N / Math.Pow(2, 24)));
                    var referenceBin = Dft.Row(i).SubArray(fn - 3 * cn, 2 * cn).Zip(Dft.Row(i).SubArray(fn + cn, 2 * cn), (s1, s2) => 0.5 * (s1.Magnitude + s2.Magnitude)).ToArray();
                    for (int k = 0; k < 2 * cn; k++)
                    {
                        Nft[i, fn - cn + k] = Dft[i, fn - cn + k] * (1 - referenceBin[k] / Dft[i, fn - cn + k].Magnitude);
                        Nft[i, N - fn + cn - k] = Complex.Conjugate(Dft[i, fn - cn + k]) * (1 - referenceBin[k] / Dft[i, fn - cn + k].Magnitude);
                    }
                }

            }

            Parallel.For(0, Ch, s => Dfil.SetRow(s, data.Row(s).Subtract(IFFT_1D(Nft.Row(s)).Select(val => (val.Real) / N).ToArray())));

            return Dfil.SubArray(0, Ch, 0, originalLength);
        }
        
    }
}
