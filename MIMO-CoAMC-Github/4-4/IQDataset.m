clc
close all 
clear all
rx_antenna=4;
tx_antenna=4;
N=128;
load('H.mat')
%train data
mode = 'train';
samples = 20000;
CreatDataset(H,samples,rx_antenna, tx_antenna, N, mode);
%test data
mode = 'test';
samples = 10000;
CreatDataset(H,samples,rx_antenna, tx_antenna, N, mode);
function []=CreatDataset(H, samples, rx_antenna, tx_antenna, N, mode)
    for SNR = -10:2:10
        filename = ['Dataset/IQ/Original/',mode,'/',num2str(SNR)];
        % 2PSK signal
        IQ_2PSK = [];
        for s = 1:samples
            x=randi([0,1],1,N);
            x = pskmod(x,2);
            x = x./std(x);
            tx_symbols = reshape(x, tx_antenna, []);
            Y = Noise(tx_symbols, H, SNR, rx_antenna);
            IQ_2PSK = [IQ_2PSK;Y];
        end
        % 4PSK signal        
       IQ_4PSK = [];
        for s = 1:samples
            x=randi([0,3],1,N);
            x = pskmod(x,4);
            x = x./std(x);
            tx_symbols = reshape(x, tx_antenna, []);
            Y = Noise(tx_symbols, H, SNR, rx_antenna);
            IQ_4PSK = [IQ_4PSK;Y];
        end
        % 8PSK signal        
       IQ_8PSK = [];
        for s = 1:samples
            x=randi([0,7],1,N);
            x = pskmod(x,8);
            x = x./std(x);
            tx_symbols = reshape(x, tx_antenna, []);
            Y = Noise(tx_symbols, H, SNR, rx_antenna);
            IQ_8PSK = [IQ_8PSK;Y];
        end
        %16QAM signal
       IQ_16QAM = [];
        for s = 1:samples
            x=randi([0,15],1,N);
            x = qammod(x,16);
            x = x./std(x);
            tx_symbols = reshape(x, tx_antenna, []);
            Y = Noise(tx_symbols, H, SNR, rx_antenna);
            IQ_16QAM = [IQ_16QAM;Y];
        end
        IQ = [IQ_2PSK; IQ_4PSK; IQ_8PSK; IQ_16QAM];
        save(filename, 'IQ')
    end
end
function Y= Noise(x,H,SNR, rxAntennas)
if nargin < 3
    rxAntennas = 2;
end

if ~isscalar(rxAntennas) || floor(rxAntennas) ~= rxAntennas || ...
        rxAntennas <= 0
    error('mimoChannel:invalidInput', ...
        'RXANTENNAS must be a positive interger');
end

if ~isscalar(SNR)
    error('mimoChannel:invalidInput', 'SNR must be a scalar');
end

% Add the noise
Es = mean(abs(x(:)).^2); % should be 1 if the constellation has been normalized
Y = awgn(H * x, SNR, 10*log10(Es));
end