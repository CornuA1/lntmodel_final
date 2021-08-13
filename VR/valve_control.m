function valve_control()
    mc = serial('COM4');
    set(mc,'BaudRate',115200);
    %set(mc,'TimeOut', 0.001);
    fopen(mc);
    pause(2);
    v_open = 0;
    k = input('r...open, t...close, b...one reward bolus, x...exit\n','s');
    while ~strcmpi(k,'x')
        if strcmpi(k,'r')
            fprintf(mc, 'r');
            v_open = 1;
        end
        if strcmpi(k,'t')
            fprintf(mc, 't');
            v_open = 0;
        end
        if strcmpi(k,'b') && v_open == 0
            fprintf(mc, 'r');
            pause(0.3);
            fprintf(mc, 't');
        end
        
        k = input('','s');
    end
    fclose(mc);
    delete(mc);
end
