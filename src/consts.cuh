#define MAX(a,b)        ((a) > (b) ? (a) : (b))
#define MIN(a,b)        ((a) < (b) ? (a) : (b))
#define ABS(a)          ((a) > 0.0 ? (a) : -(a))

#define True            1
#define False           0

#define INF             2e19
#define EPS             1e-10
#define ISAPPROX(a,b)   ( ABS( (a) - (b) ) <= EPS * MAX( ABS(a), ABS(b) ) )

#define LABEL_LEN       128
#define ID_LEN          4
#define BUS_BRANCH_LEN  11

#define FORTRAN_START_INDEX 1

#define NORMTYPE_L1    1
#define NORMTYPE_L2    2
#define NORMTYPE_LINF  3

typedef enum {
    Ctg_Gen = 0,
    Ctg_Branch,
} CtgType;

typedef enum {
    Network_Output_Full = 0,
    Network_Output_Summary
} NetworkOutputOption;

typedef enum {
    Network_Input_Matpower = 0,
    Network_Input_Go,
} NetworkInputType;

typedef enum {
    Objective_PiecewiseLinear = 1,
    Objective_Quadratic,
} ObjectiveType;

typedef enum {
    Solve_Succeed = 0,
    Solve_Acceptable,
    Solve_Infeasible,
    Solve_IterLimit,
    Solve_TimeLimit,
    Solve_RestorationFailed,
    Solve_InvalidOption,
    Solve_InsufficientMemory,
    Solve_OtherError
} SolverStatus;

typedef enum {
    OK                              =  0,
    Error_NullPointer               = -1,
    Error_NotFound                  = -2,
    Error_InvalidContingency        = -3,
    Error_InvalidFileFormat         = -4,
    Error_InvalidLPMethod           = -5,
    Error_InvalidOption             = -6,
    Error_InvalidOptionType         = -7,
    Error_InvalidPowerFlow          = -8,
    Error_InvalidSize               = -9,
    Error_InvalidSolver             = -10,
    Error_SolverFunctionFailed      = -11,
    Error_SolverNotInitialized      = -12,
    Error_UnsupportedConstraint     = -13,
    Error_UnsupportedFunction       = -14,
    Error_TooSmall                  = -15,
    Error_TooLarge                  = -16,
    Error_MaximumReached            = -17,
} ExitStatus;
