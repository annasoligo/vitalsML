audioFile = 'gut2.wav';
output_filepath = "./scalograms/gut/";

[A,fs] = audioread(audioFile);

for i = 1:length(A)
    if abs(A(i)) < 0.5
        A(i) = 0;
    end    
end

chunk = fs*0.03;
lenA = length(A);
sampleLen = lenA - mod(lenA, chunk);
n = [0];
for i= chunk:lenA-chunk
    if A(i)>0.95
        if i> (n(end)+ chunk/2)
            n(end+1) = i;
        end    
    end
end 

for i = 2:length(n)

    audio = A(n(i)-chunk/2: n(i)+chunk/2, :);
    figure(1);
    CWTcoeffs = cwt(audio,5:5:500,'sym10','plot');  %cwt for continuous 1D wavelet transform
    colormap gray; 
    %colorbar;
    %title('Far Nodule')
    %xlabel('Sample (time)','fontsize',16);
    %ylabel('Frequency','fontsize',16)
    
    %uncommenting the below lines will save the plots and overwrite the
    %existing scalograms, use carefully

    set(gca, 'Visible', 'off');
    saveas(gcf,(output_filepath + 'g1_' + num2str(i) + ".png"))   %save figure to output filepath
    close
end
