function [mat1out, mat2out] = splitmat(mat)
leng1 = size(mat,1);
run = 0;
while run < leng1
    if run == 0
        mat1out = mat(1:3600,:);
        mat2out = mat(3601:4000,:);
        run = 4000;
    elseif leng1 - run <= 4000
        mat2out = cat(1,mat2out,mat(run:leng1,:));
        run = run + 4000;
        break
    else
        mat1out = cat(1,mat1out,mat(run+1:run+3600,:));
        mat2out = cat(1,mat2out,mat(run+3601:run+4000,:));
        run = run + 4000;
    end
end
end