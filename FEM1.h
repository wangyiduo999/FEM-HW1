/*This is a template file for use with 1D finite elements.
  The portions of the code you need to fill in are marked with the comment "//EDIT".

  Do not change the name of any existing functions, but feel free
  to create additional functions, variables, and constants.
  It uses the deal.II FEM library.*/

//Include files
//Data structures and solvers
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
//Mesh related classes
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
//Finite element implementation classes
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
//Standard C++ libraries
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdint.h>
#include <cfloat>

using namespace std;
using namespace dealii;

template <int dim>
class FEM
{
public:
  //Class functions
  FEM (unsigned int order, unsigned int problem); // Class constructor
  ~FEM(); //Class destructor

  //Function to find the value of xi at the given node (using deal.II node numbering)
  double xi_at_node(unsigned int dealNode);

  //Define your 1D basis functions and derivatives
  double basis_function(unsigned int node, double xi);
  double basis_gradient(unsigned int node, double xi);

  //Solution steps
  void generate_mesh(unsigned int numberOfElements);
  void define_boundary_conds();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results();
  double analytical_solution_for_problem(double x);

  //Function to calculate the l2 norm of the error in the finite element sol'n vs. the exact solution
  double l2norm_of_error();

  //Class objects
  Triangulation<dim>   triangulation; //mesh
  FESystem<dim>        fe;        //FE element
  DoFHandler<dim>      dof_handler;   //Connectivity matrices

  //Gaussian quadrature - These will be defined in setup_system()
  unsigned int          quadRule;    //quadrature rule, i.e. number of quadrature points
  std::vector<double> quad_points; //vector of Gauss quadrature points
  std::vector<double> quad_weight; //vector of the quadrature point weights

  //Data structures
  SparsityPattern       sparsity_pattern; //Sparse matrix pattern
  SparseMatrix<double>  K;     //Global stiffness (sparse) matrix
  Vector<double>        D, F;      //Global vectors - Solution vector (D) and Global force vector (F)
  std::vector<double>   nodeLocation;  //Vector of the x-coordinate of nodes by global dof number
  std::map<unsigned int, double> boundary_values; //Map of dirichlet boundary conditions
  double                basisFunctionOrder, prob, L, g1, g2;
  double                body_f, E, Area, h2;

  //solution name array
  std::vector<std::string> nodal_solution_names;
  std::vector<DataComponentInterpretation::DataComponentInterpretation> nodal_data_component_interpretation;
};

// Class constructor for a vector field
template <int dim>
FEM<dim>::FEM(unsigned int order, unsigned int problem)
  :
  fe (FE_Q<dim>(order), dim),
  dof_handler (triangulation)
{
  basisFunctionOrder = order;
  if (problem == 1 || problem == 2) {
    prob = problem;
  }
  else {
    std::cout << "Error: problem number should be 1 or 2.\n";
    exit(0);
  }

  //Nodal Solution names - this is for writing the output file
  for (unsigned int i = 0; i < dim; ++i) {
    nodal_solution_names.push_back("u");
    nodal_data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
  }
}

//Class destructor
template <int dim>
FEM<dim>::~FEM() {
  dof_handler.clear();
}

//Find the value of xi at the given node (using deal.II node numbering)
template <int dim>
double FEM<dim>::xi_at_node(unsigned int dealNode) {
  double xi;

  if (dealNode == 0) {
    xi = -1.;
  }
  else if (dealNode == 1) {
    xi = 1.;
  }
  else if (dealNode <= basisFunctionOrder) {
    xi = -1. + 2.*(dealNode - 1.) / basisFunctionOrder;
  }
  else {
    std::cout << "Error: you input node number "
              << dealNode << " but there are only "
              << basisFunctionOrder + 1 << " nodes in an element.\n";
    exit(0);
  }

  return xi;
}

//Define basis functions
template <int dim>
double FEM<dim>::basis_function(unsigned int node, double xi) {
  /*"basisFunctionOrder" defines the polynomial order of the basis function,
    "node" specifies which node the basis function corresponds to,
    "xi" is the point (in the bi-unit domain) where the function is being evaluated.
    You need to calculate the value of the specified basis function and order at the given quadrature pt.*/

  // if (!(node >= 0 && node <= basisFunctionOrder && xi >= -1 && xi <= 1)) {
  //   cout << "node: " << node <<endl;
  //   cout << "xi:" <<xi <<endl;

  // }
  // assert(node >= 0 && node <= basisFunctionOrder && xi >= -1 && xi <= 1);
  int node_numbers = basisFunctionOrder + 1;
  std::vector<double> xi_b_nodes;
  double xi_a_node = xi_at_node(node);

  for (int i = 0; i < node_numbers; i++) {
    xi_b_nodes.push_back(xi_at_node(i));
  }

  // double nominator = 1.;
  // double denominator = 1.;
  // cout << xi_a_node <<endl;
  int count = 0;
  double value = 1.0;
  for (int i = 0; i < node_numbers; i++) {
    // cout << xi_b_nodes[i] <<endl;
    if (i != node) {
      // nominator *= (xi - xi_b_nodes[i]);
      // denominator *= (xi_a_node - xi_b_nodes[i]);
      value *= (xi - xi_b_nodes[i]) / (xi_a_node - xi_b_nodes[i]);
      count++;
    }
  }

  if (count != node_numbers - 1) {
    cout << "error" << endl;
    assert("error");
  }
  // getchar();

  // cout << "nominator = " << nominator << endl;
// cout << "denominator = " << denominator << endl;
//  double value = nominator / denominator; //Store the value of the basis function in this variable
// printf("node = %d, xi = %lf, value = %lf\n", node, xi, value);

  /*You can use the function "xi_at_node" (defined above) to get the value of xi (in the bi-unit domain)
    at any node in the element - using deal.II's element node numbering pattern.*/

  //EDIT

  return value;
}

//Define basis function gradient
template <int dim>
double FEM<dim>::basis_gradient(unsigned int node, double xi) {
  /*"basisFunctionOrder" defines the polynomial order of the basis function,
    "node" specifies which node the basis function corresponds to,
    "xi" is the point (in the bi-unit domain) where the function is being evaluated.
    You need to calculate the value of the derivative of the specified basis function and order at the given quadrature pt.
    Note that this is the derivative with respect to xi (not x)*/
  //double delta_xi = FLT_MIN * 1000000;
  static const double delta_xi = 1.0e-7;
// static const double delta_xi = 0.0000000000001;
  double derivative = (basis_function(node, xi + delta_xi) - basis_function(node, xi)) / delta_xi;
  // double derivative_check = (-1.0) * (basis_function(node, xi - delta_xi) - basis_function(node, xi)) / delta_xi;
  // double diff = derivative - derivative_check;
  // if (derivative != -derivative_check) {
  //   cout << "derivative error" << endl;
  //   printf("derivative = %.20f\n", derivative);
  //   printf("diff == %.20f\n", diff);
  //  // getchar();
  //   //return 0.0;
  // }

  if (basisFunctionOrder == 1) {
    if (node == 0)
      return 0.5;
    else if (node == 1)
      return -0.5;
  } else if (basisFunctionOrder == 2) {
    if (node == 0)
      return -0.5 + xi;
    else if (node == 1)
      return 0.5 + xi;
    else
      return -2.0 * xi;
  } else if (basisFunctionOrder == 3) {
    switch (node) {
    case 0:
      return (-27.0 * xi * xi) / (16.0) + 9.0 * xi / 8.0 + 1.0 / 16.0;
    case 2:
      return (81.0 * xi * xi) / (16.0) - 9.0 * xi / 8.0 - 27.0 / 16.0;
    case 3:
      return -(81.0 * xi * xi) / (16.0) - 9.0 * xi / 8.0 + 27.0 / 16.0;
    case 1:
      return (27.0 * xi * xi) / (16.0) + 9.0 * xi / 8.0 - 1.0 / 16.0;
    }
  }


// cout << setprecision(10) <<"derivative at node:" << node << " xi: " << xi << " value: " << derivative << endl;
  //double value = derivative; //Store the value of the gradient of the basis function in this variable
  double value = derivative;
  /*You can use the function "xi_at_node" (defined above) to get the value of xi (in the bi-unit domain)
    at any node in the element - using deal.II's element node numbering pattern.*/

  return value;
}

//Define the problem domain and generate the mesh
template <int dim>
void FEM<dim>::generate_mesh(unsigned int numberOfElements) {

  //Define the limits of your domain
  L = 0.1; //EDIT
  double x_min = 0.;
  double x_max = L;

  Point<dim, double> min(x_min),
        max(x_max);
  std::vector<unsigned int> meshDimensions (dim, numberOfElements);
  GridGenerator::subdivided_hyper_rectangle (triangulation, meshDimensions, min, max);
}

//Specify the Dirichlet boundary conditions
template <int dim>
void FEM<dim>::define_boundary_conds() {
  const unsigned int totalNodes = dof_handler.n_dofs(); //Total number of nodes

  //Identify dirichlet boundary nodes and specify their values.
  //This function is called from within "setup_system"

  /*The vector "nodeLocation" gives the x-coordinate in the real domain of each node,
    organized by the global node number.*/

  /*This loops through all nodes in the system and checks to see if they are
    at one of the boundaries. If at a Dirichlet boundary, it stores the node number
    and the applied displacement value in the std::map "boundary_values". Deal.II
    will use this information later to apply Dirichlet boundary conditions.
    Neumann boundary conditions are applied when constructing Flocal in "assembly"*/
  for (unsigned int globalNode = 0; globalNode < totalNodes; globalNode++) {
    if (nodeLocation[globalNode] == 0) {
      boundary_values[globalNode] = g1;
    }
    if (nodeLocation[globalNode] == L) {
      if (prob == 1) {
        boundary_values[globalNode] = g2;
      }
    }
  }

}

//Setup data structures (sparse matrix, vectors)
template <int dim>
void FEM<dim>::setup_system() {

  //Define constants for problem (Dirichlet boundary values)
  g1 = 0; g2 = 0.001; //EDIT

  //Let deal.II organize degrees of freedom
  dof_handler.distribute_dofs (fe);

  //Enter the global node x-coordinates into the vector "nodeLocation"
  MappingQ1<dim, dim> mapping;
  std::vector< Point<dim, double> > dof_coords(dof_handler.n_dofs());
  nodeLocation.resize(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points<dim, dim>(mapping, dof_handler, dof_coords);
  for (unsigned int i = 0; i < dof_coords.size(); i++) {
    nodeLocation[i] = dof_coords[i][0];
  }

  //Specify boundary condtions (call the function)
  define_boundary_conds();

  //Define the size of the global matrices and vectors
  sparsity_pattern.reinit (dof_handler.n_dofs(), dof_handler.n_dofs(),
                           dof_handler.max_couplings_between_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
  sparsity_pattern.compress();
  K.reinit (sparsity_pattern);
  F.reinit (dof_handler.n_dofs());
  D.reinit (dof_handler.n_dofs());

  //Define quadrature rule
  /*A quad rule of 2 is included here as an example. You will need to decide
    what quadrature rule is needed for the given problems*/

  if (basisFunctionOrder == 0) {
    quadRule = 2; //EDIT - Number of quadrature points along one dimension
    quad_points.resize(quadRule); quad_weight.resize(quadRule);

    quad_points[0] = -sqrt(1. / 3.); //EDIT
    quad_points[1] = sqrt(1. / 3.); //EDIT

    quad_weight[0] = 1.; //EDIT
    quad_weight[1] = 1.; //EDIT
  } else if ( basisFunctionOrder == 1) {
    quadRule = 3; //EDIT - Number of quadrature points along one dimension
    quad_points.resize(quadRule); quad_weight.resize(quadRule);

    quad_points[0] = -sqrt(3.0 / 5.0); //EDIT
    quad_points[1] = 0; //EDIT
    quad_points[2] = sqrt(3.0 / 5.0);

    quad_weight[0] = 5.0 / 9.0; //EDIT
    quad_weight[1] = 8.0 / 9.0; //EDIT
    quad_weight[2] = 5.0 / 9.0; //EDIT


  } else if (basisFunctionOrder == 2 ) {
    // cout << "basis function order == 3" <<endl;
    quadRule = 4; //EDIT - Number of quadrature points along one dimension
    quad_points.resize(quadRule); quad_weight.resize(quadRule);

    quad_points[0] = -sqrt(3.0 / 7.0 + (2.0 / 7.0) * sqrt(6.0 / 5.0)); //EDIT
    quad_points[1] = -sqrt(3.0 / 7.0 - (2.0 / 7.0) * sqrt(6.0 / 5.0)); //EDIT
    quad_points[2] = sqrt(3.0 / 7.0 - (2.0 / 7.0) * sqrt(6.0 / 5.0));
    quad_points[3] = sqrt(3.0 / 7.0 + (2.0 / 7.0) * sqrt(6.0 / 5.0));

    quad_weight[0] = (18.0 - sqrt(30.0)) / 36.0; //EDIT
    quad_weight[1] = (18.0 + sqrt(30.0)) / 36.0; //EDIT
    quad_weight[2] = (18.0 + sqrt(30.0)) / 36.0;
    quad_weight[3] = (18.0 - sqrt(30.0)) / 36.0;
  } else if (false) {
    quadRule = 5; //EDIT - Number of quadrature points along one dimension
    quad_points.resize(quadRule); quad_weight.resize(quadRule);

    quad_points[0] = 0.0;
    quad_points[1] = -(1.0 / 3.0) * sqrt(5.0 - 2.0 * sqrt(10.0 / 7.0)); //EDIT
    quad_points[2] = +(1.0 / 3.0) * sqrt(5.0 - 2.0 * sqrt(10.0 / 7.0));
    quad_points[3] = -(1.0 / 3.0) * sqrt(5.0 + 2.0 * sqrt(10.0 / 7.0));
    quad_points[4] = +(1.0 / 3.0) * sqrt(5.0 + 2.0 * sqrt(10.0 / 7.0));

    quad_weight[0] = 128.0 / 225.0;
    quad_weight[1] = (322.0 + 13.0 * sqrt(70.0)) / 900.0; //EDIT
    quad_weight[2] = (322.0 + 13.0 * sqrt(70.0)) / 900.0; //EDIT
    quad_weight[3] = (322.0 - 13.0 * sqrt(70.0)) / 900.0;
    quad_weight[4] = (322.0 - 13.0 * sqrt(70.0)) / 900.0;
  } else if (false) {
    quadRule = 7; //EDIT - Number of quadrature points along one dimension
    quad_points.resize(quadRule); quad_weight.resize(quadRule);

    quad_points[0] = 0.0;
    quad_points[1] = -sqrt(5.0 / 11.0 - (2.0 / 11.0) * sqrt(5.0 / 3.0)); //EDIT
    quad_points[2] = +sqrt(5.0 / 11.0 - (2.0 / 11.0) * sqrt(5.0 / 3.0));
    quad_points[3] = -sqrt(5.0 / 11.0 + (2.0 / 11.0) * sqrt(5.0 / 3.0));
    quad_points[4] = +sqrt(5.0 / 11.0 + (2.0 / 11.0) * sqrt(5.0 / 3.0));
    quad_points[5] = 1.0;
    quad_points[6] = -1.0;

    quad_weight[0] = 256.0 / 525.0;
    quad_weight[1] = (124.0 + 7.0 * sqrt(15.0)) / 350.0; //EDIT
    quad_weight[2] = (124.0 + 7.0 * sqrt(15.0)) / 350.0;
    quad_weight[3] = (124.0 - 7.0 * sqrt(15.0)) / 350.0;
    quad_weight[4] = (124.0 - 7.0 * sqrt(15.0)) / 350.0;
    quad_weight[5] = 1.0 / 21.0;
    quad_weight[6] = 1.0 / 21.0;
    // quad_points[0] = 0.0000000000000000;
    // quad_points[1] = 0.4058451513773972; //EDIT
    // quad_points[2] = -0.4058451513773972;
    // quad_points[3] = -0.7415311855993945;
    // quad_points[4] = 0.7415311855993945;
    // quad_points[5] = -0.9491079123427585;
    // quad_points[6] = 0.9491079123427585;

    // quad_weight[0] = 0.4179591836734694;
    // quad_weight[1] = 0.3818300505051189; //EDIT
    // quad_weight[2] = 0.3818300505051189; //EDIT
    // quad_weight[3] = 0.2797053914892766;
    // quad_weight[4] = 0.2797053914892766;
    // quad_weight[5] = 0.1294849661688697;
    // quad_weight[6] = 0.1294849661688697;
  } else if (basisFunctionOrder == 3) {
    quadRule = 10; //EDIT - Number of quadrature points along one dimension
    quad_points.resize(quadRule); quad_weight.resize(quadRule);

    quad_points[0] = -0.1488743389816312108848260011297199846;
    quad_points[1] = 0.1488743389816312108848260011297199846; //EDIT
    quad_points[2] = -0.4333953941292471907992659431657841622;
    quad_points[3] = 0.4333953941292471907992659431657841622;
    quad_points[4] = -0.6794095682990244062343273651148735758;
    quad_points[5] = 0.6794095682990244062343273651148735758;
    quad_points[6] = -0.8650633666889845107320966884234930485;
    quad_points[7] = 0.8650633666889845107320966884234930485;
    quad_points[8] = -0.9739065285171717200779640120844520534;
    quad_points[9] = 0.9739065285171717200779640120844520534;


    quad_weight[0] = 0.295524224714752870173892994651338329;
    quad_weight[1] = 0.295524224714752870173892994651338329; //EDIT
    quad_weight[2] = 0.269266719309996355091226921569469352;
    quad_weight[3] = 0.269266719309996355091226921569469352;
    quad_weight[4] = 0.219086362515982043995534934228163192;
    quad_weight[5] = 0.219086362515982043995534934228163192;
    quad_weight[6] = 0.149451349150580593145776339657697332;
    quad_weight[7] = 0.149451349150580593145776339657697332;
    quad_weight[8] = 0.066671344308688137593568809893331792;
    quad_weight[9] = 0.066671344308688137593568809893331792;
  }


// for (int i = 0; i < quad_points.size(); i++) {
//    printf( "quad_points[%d] = %.10f, quad_weight[%d] = %.10f\n", i, quad_points[i], i, quad_weight[i]);
// }

//Just some notes...
  std::cout << "   Number of active elems:       " << triangulation.n_active_cells() << std::endl;
  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
}

//Form elmental vectors and matrices and assemble to the global vector (F) and matrix (K)
template <int dim>
void FEM<dim>::assemble_system() {

  K = 0; F = 0;

  const unsigned int        dofs_per_elem = fe.dofs_per_cell; //This gives you number of degrees of freedom per element
  FullMatrix<double>        Klocal (dofs_per_elem, dofs_per_elem);
  Vector<double>            Flocal (dofs_per_elem);
  std::vector<unsigned int> local_dof_indices (dofs_per_elem);
  double                    h_e, x, f;



  body_f = pow(10, 11);
  f = body_f;
  E = pow(10, 11);
  Area = pow(10, -4);
  h2 = pow(10, 6);


  //cout <<"Area" << Area <<endl;
  //loop over elements
  typename DoFHandler<dim>::active_cell_iterator elem = dof_handler.begin_active(),
                                                 endc = dof_handler.end();
  for (; elem != endc; ++elem) {

    /*Retrieve the effective "connectivity matrix" for this element
      "local_dof_indices" relates local dofs to global dofs,
      i.e. local_dof_indices[i] gives the global dof number for local dof i.*/
    elem->get_dof_indices (local_dof_indices);

    /*We find the element length by subtracting the x-coordinates of the two end nodes
      of the element. Remember that the vector "nodeLocation" holds the x-coordinates, indexed
      by the global node number. "local_dof_indices" gives us the global node number indexed by
      the element node number.*/
    h_e = nodeLocation[local_dof_indices[1]] - nodeLocation[local_dof_indices[0]];
    //h_e = h_e/10;
    //out_h_e =h_e;
    // cout << "h_e:" << h_e << endl;
    //Loop over local DOFs and quadrature points to populate Flocal and Klocal.
    Flocal = 0.;

    for (int i = 0; i < dofs_per_elem; i++) {
      Flocal[i] = 0;
    }

    for (unsigned int A = 0; A < dofs_per_elem; A++) {
      for (unsigned int q = 0; q < quadRule; q++) {
        x = 0;
        //Interpolate the x-coordinates at the nodes to find the x-coordinate at the quad pt. Yiduo: Calculate x
        for (unsigned int B = 0; B < dofs_per_elem; B++) {
          x += nodeLocation[local_dof_indices[B]] * basis_function(B, quad_points[q]);

        }
        //EDIT - Define Flocal.
        // cout << "x:" << x <<endl;
        Flocal[A] += Area * h_e * (f * x) * basis_function(A, quad_points[q]) * quad_weight[q] / 2;
      }
      //printf("Flocal[%d] = %.10f\n", A, 2*Flocal[A]/(f*h_e*nodeLocation[A_global]));
    }
    //Add nonzero Neumann condition, if applicable
    if (prob == 2) {
      if (nodeLocation[local_dof_indices[1]] == L) {
        //EDIT - Modify Flocal to include the traction on the right boundary.


        double t = h2;
        Flocal[1] += t;
        // printf("Flocal[%d] = %.10f\n", dofs_per_elem - 1, Flocal[dofs_per_elem - 1]);
      }
    }

    //Loop over local DOFs and quadrature points to populate Klocal

    for (int i = 0; i < dofs_per_elem; i++) {
      for (int j = 0; j < dofs_per_elem; j++)
        Klocal[i][j] = 0;
    }



    for (unsigned int A = 0; A < dofs_per_elem; A++) {
      for (unsigned int B = 0; B < dofs_per_elem; B++) {
        for (unsigned int q = 0; q < quadRule; q++) {
          //EDIT - Define Klocal.
          Klocal[A][B] +=  Area * (E / h_e ) * (2 * basis_gradient(A, quad_points[q]) *
                                                (basis_gradient(B, quad_points[q]) * quad_weight[q])) ;
//          cout << "A" <<"B" <<"Klocal"Klocal[A][B] <<endl;
        }
        // printf("Klocal[%d][%d] = %lf\n", A, B, Klocal[A][B]);
      }
    }

    //Assemble local K and F into global K and F
    //You will need to used local_dof_indices[A]
    for (unsigned int A = 0; A < dofs_per_elem; A++) {
      //EDIT - add component A of Flocal to the correct location in F
      /*Remember, local_dof_indices[A] is the global degree-of-freedom number
      corresponding to element node number A*/
      unsigned int A_global = local_dof_indices[A];
      F[A_global] += Flocal[A];
      //  printf("Global[%d] = %.10f\n", A_global, 2*F[A_global]/(f*h_e*nodeLocation[A_global]));
      for (unsigned int B = 0; B < dofs_per_elem; B++) {
        //EDIT - add component A,B of Klocal to the correct location in K (using local_dof_indices)
        /*Note: K is a sparse matrix, so you need to use the function "add".
          For example, to add the variable C to K[i][j], you would use:
          K.add(i,j,C);*/
        unsigned int B_global = local_dof_indices[B];
        K.add(A_global, B_global, Klocal[A][B]);
        //    printf("Kglobal[%d][%d] = %lf\n", A_global, B_global, Klocal[A][B]*h_e / (E * Area * 2));
      }
    }

  }

  // for (int i = 0; i < F.size(); i++) {
  //   for (int j = 0; j < F.size(); j++)
  //  // printf("K[%d][%d] = %lf\n", i, j, K.el(i,j));
  // //printf("K[%d][%d] = %lf\n", i, j, K.el(i,j)*out_h_e/E);
  // }

  //Apply Dirichlet boundary conditions
  /*deal.II applies Dirichlet boundary conditions (using the boundary_values map we
    defined in the function "define_boundary_conds") without resizing K or F*/
  MatrixTools::apply_boundary_values (boundary_values, K, D, F, false);
}

//Solve for D in KD=F
template <int dim>
void FEM<dim>::solve() {

  //Solve for D
  SparseDirectUMFPACK  A;
  A.initialize(K);
  A.vmult (D, F); //D=K^{-1}*F

}

//Output results
template <int dim>
void FEM<dim>::output_results () {

  //Write results to VTK file
  std::ofstream output1("solution.vtk");
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  //Add nodal DOF data
  data_out.add_data_vector(D, nodal_solution_names, DataOut<dim>::type_dof_data,
                           nodal_data_component_interpretation);
  data_out.build_patches();
  data_out.write_vtk(output1);
  output1.close();
}

template <int dim>
double FEM<dim>::analytical_solution_for_problem(double x) {

  if (prob == 1) {

    // cout << "body_f:" << body_f <<endl;

    // cout << "(-1.0) * x * x * x *body_f :" << (-1.0) * x * x * x *body_f << endl;
    // getchar();
    // cout << " (6.0 * E)" <<  (6.0 * E) << endl;
    // getchar();
    // cout << "first part " << (-1.0) * x * x * x *body_f / (6.0 * E) << endl;
    // getchar();
    // cout << x * ((g2 - g1) / L + body_f * L *L/ (6.0 * E )) <<endl;
    // getchar();
    // cout << "anyltical result123 = " << result <<endl;
    return (-1.0) * x * x * x * body_f / (6.0 * E) + x * ((g2 - g1) / L + body_f * L * L / (6.0 * E )) + g1;
  } else {
    return (-1.0) * x * x * x * body_f / (6.0 * E) + x * (( 0.5 * L * L * body_f * Area + h2) / (E * Area)) + g1;
  }


}

template <int dim>
double FEM<dim>::l2norm_of_error() {

  double l2norm = 0.0;

  // double xi_end = analytical_solution_for_problem(L);
  // double xi_begin = analytical_solution_for_problem(0);
  // cout << "analytical_solution_for_problem : " << xi_begin << endl;
  // cout << "analytical_solution_for_problem : " << xi_end << endl;
  //Find the l2 norm of the error between the finite element sol'n and the exact sol'n
  const unsigned int        dofs_per_elem = fe.dofs_per_cell; //This gives you dofs per element
  std::vector<unsigned int> local_dof_indices (dofs_per_elem);
  double u_exact, u_h, x, h_e;

  //loop over elements

//   for (int i = 0; i < D.size(); i++) {
//     // printf("D[%d] = %.10f anyltical[%lf] = %.10f, difference = %.10f\n", i,
//            // D[i], nodeLocation[i] , analytical_solution_for_problem(nodeLocation[i]), (D[i] - analytical_solution_for_problem(nodeLocation[i])));
// //   cout << (D[i] - analytical_solution_for_problem(i * 0.01)) << endl;
//   }
//  cout << "numberical : " << D[0] << endl;
// cout << "numberical : " << D[D.size()-1] << endl;


  // for (int i = 0; i < D.size(); i++) {
  //   // sscanf(%)

  // }


  typename DoFHandler<dim>::active_cell_iterator elem = dof_handler.begin_active (),
                                                 endc = dof_handler.end();
  for (; elem != endc; ++elem) {

    //Retrieve the effective "connectivity matrix" for this element
    elem->get_dof_indices (local_dof_indices);

    //Find the element length
    h_e = nodeLocation[local_dof_indices[1]] - nodeLocation[local_dof_indices[0]];

    for (unsigned int q = 0; q < quadRule; q++) {
      x = 0.; u_h = 0.;
      //Find the values of x and u_h (the finite element solution) at the quadrature points
      for (unsigned int B = 0; B < dofs_per_elem; B++) {
        x += nodeLocation[local_dof_indices[B]] * basis_function(B, quad_points[q]);
        u_h += D[local_dof_indices[B]] * basis_function(B, quad_points[q]);
      }
      //EDIT - Find the l2-norm of the error through numerical integration.
      /*This includes evaluating the exact solution at the quadrature points*/
      u_exact = analytical_solution_for_problem(x);

      // printf("x = %.25f, u_exact = %.25f, u_h = %.25f, diff = %.25f\n", x, u_exact, u_h, abs(u_exact - u_h));

      l2norm += (u_exact - u_h) * (u_exact - u_h) * h_e * quad_weight[q] / 2.0;


    }
  }
  cout << "l2norm: " << sqrt(l2norm) << "sqaure:" << l2norm << endl;
  return sqrt(l2norm);
}
