function plot_spec(sig, fs)

% Function to plot frequency spectrum of sig
%   usage: 
%           plot_spectrum(sig, 1000)
%
% author: Peter YK Cheung, 9 Jan 2019
    magnitude = abs(fft(sig));
    N = length(sig);
    df = fs/N; 
    f = 0:df:fs/2;
    Y = magnitude(1:length(f));
    plot(f, 2*Y/N)
    xlabel('\fontsize{14}frequency (Hz)')
    ylabel('\fontsize{14}Magnitude');
    title('\fontsize{16}Spectrum');
    