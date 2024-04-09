clc 
close all 
clear all
rx_antenna=4;
tx_antenna=1;
%Constant Channel Matrix
H = randn(rx_antenna, tx_antenna) / sqrt(2) +...
     1j * randn(rx_antenna, tx_antenna) / sqrt(2);
save('H', 'H');
function H_CC = ChannelCorrelation(rho_abs, rx_antenna,tx_antenna, H)
rho_int = rand()+1j*rand();
rho_int_abs = abs(rho_int);
rho = rho_int/rho_int_abs*rho_abs;
R_rx=zeros([rx_antenna,rx_antenna]);
for i = 1:rx_antenna
    for j = 1:rx_antenna
        if i<=j
            R_rx(i,j) = rho^(j-i);
        else
            R_rx(i,j) = conj(R_rx(j,i));
        end
    end
end

R_tx=zeros([tx_antenna,tx_antenna]);
for i = 1:tx_antenna
    for j = 1:tx_antenna
        if i<=j
            R_tx(i,j) = rho^(j-i);
        else
            R_tx(i,j) = conj(R_tx(j,i));
        end
    end
end
H_CC = sqrt(R_rx)*H*sqrt(R_tx);
end