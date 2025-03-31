function settings = load_test_case(case_no)

switch case_no

    case 1

        settings.mu_x = 0;
        settings.C_xx = 1;
        settings.mu_q = 0;
        settings.C_qq = 0.25;
        settings.mu_d = 0;
        settings.C_d  = 1;

        settings.beta = 0.3;
        settings.model_type = 1;

    case 2

        settings.mu_x = 1;
        settings.C_xx = 1;
        settings.mu_q = 0;
        settings.C_qq = 0.25;
        settings.mu_d = -1;
        settings.C_d  = 1;

        settings.beta = 0.3;
        settings.model_type = 1;


    case 3

        settings.mu_x = 1;
        settings.C_xx = 1;
        settings.mu_q = 0;
        settings.C_qq = 0.25;
        settings.mu_d = 0;
        settings.C_d  = 1;

        settings.beta = 0.3;
        settings.model_type = 2;

    case 4

        settings.mu_x = 1;
        settings.C_xx = 1;
        settings.mu_q = 0;
        settings.C_qq = 0.25;
        settings.mu_d = 0;
        settings.C_d  = 1;

        settings.beta = 0.3;
        settings.model_type = 0; % linear

    otherwise
        error('test case not existing');


end

