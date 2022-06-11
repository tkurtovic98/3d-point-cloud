function score = ThreeD_MatchRMS_Score(gt, gt_info, result)

    %function computes rms values between ground true and estimateed registration

    %for 3DMatch dataset and using 3DMatch approximate calculation

    % size(gt)
    % size(result)

    trans = gt ^ -1 * result; info = gt_info;

    te = trans( 1 : 3, 4 );

    qt = dcm2quat( trans( 1 : 3, 1 : 3 ) );

    er = [ te; - qt( 2 : 4 )' ];

    p = er' * info * er / info( 1, 1 );

    score = p;
