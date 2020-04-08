import argparse
import sys
from enum import Enum, auto

argreg1 = ["dil", "sil", "dl", "cl", "r8b", "r9b"]
argreg8 = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]

class ErrorCode(Enum):
    UNEXPECTED_TOKEN = 'Unexpected token'
    ID_NOT_FOUND     = 'Identifier not found'
    DUPLICATE_ID     = 'Duplicate id found'

class Error(Exception):
    def __init__(self, error_code=None, token=None, message=None):
        self.error_code = error_code
        self.token = token
        # add exception class name before the message
        self.message = f'{self.__class__.__name__}: {message}'

class LexerError(Error):
    pass

class ParserError(Error):
    pass

###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################

class TokenType(Enum):
    # single-character token types
    PLUS          = '+'
    MINUS         = '-'
    MUL           = '*'
    DIV           = '/'
    LPAREN        = '('
    RPAREN        = ')'
    LBRACE        = '{'
    RBRACE        = '}'
    LBRACKET      = '['
    RBRACKET      = ']'
    SEMI          = ';'
    DOT           = '.'
    COLON         = ':'
    COMMA         = ','
    LT            = '<'
    LE            = '<='
    GT            = '>'
    GE            = '>='
    NE            = '!='
    EQ            = '=='
    # block of reserved words
    INT           = 'INT'
    CHAR          = 'CHAR'
    IF            = 'IF'
    ELSE          = 'ELSE'
    WHILE         = 'WHILE'
    FOR           = 'FOR'
    NULL          = 'NULL'
    SIZEOF        = 'SIZEOF'
    STRUCT        = 'STRUCT'
    RETURN        = 'RETURN'
    # misc
    ID            = 'ID'
    INTEGER_CONST = 'INTEGER_CONST'
    REAL_CONST    = 'REAL_CONST'
    ASSIGN        = '='
    EOF           = 'EOF'
    EXPRSTMT      = 'EXPRSTMT'
    STMTEXPR      = 'STMTEXPR'
    VAR           = 'VAR'
    IFSTMT        = 'IFSTMT'
    WHILESTMT     = 'WHILESTMT'
    FORSTMT       = 'FORSTMT'
    BLOCK         = 'BLOCK'
    FUNCALL       = 'FUNCALL'
    ADDR          = '&'
    DEREF         = 'DEREF'
    PTRADD        = 'PTRADD'
    PTRSUB        = 'PTRSUB'
    PTRDIFF       = 'PTRDIFF'
    TYINT         = 'TYINT'
    TYCHAR        = 'TYCHAR'
    TYPTR         = 'TYPTR'
    TYARRAY       = 'TYARRAY'
    TYSTRUCT      = 'TYSTRUCT'
    STR           = 'STR'
    MEMBER        = 'MEMBER'

class Token:
    def __init__(self, type, value, lineno=None, column=None):
        self.type = type
        self.value = value
        self.lineno = lineno
        self.column = column

    def __str__(self):
        """String representation of the class instance.

        Example:
            >>> Token(TokenType.INTEGER, 7, lineno=5, column=10)
            Token(TokenType.INTEGER, 7, position=5:10)
        """
        return 'Token({type}, {value}, position={lineno}:{column})'.format(
            type=self.type,
            value=repr(self.value),
            lineno=self.lineno,
            column=self.column,
        )

    def __repr__(self):
        return self.__str__()


def _build_reserved_keywords():
    """Build a dictionary of reserved keywords.

    The function relies on the fact that in the TokenType
    enumeration the beginning of the block of reserved keywords is
    marked with PROGRAM and the end of the block is marked with
    the END keyword.

    Result:
        {'PROGRAM': <TokenType.PROGRAM: 'PROGRAM'>,
         'INTEGER': <TokenType.INTEGER: 'INTEGER'>,
         'REAL': <TokenType.REAL: 'REAL'>,
         'DIV': <TokenType.INTEGER_DIV: 'DIV'>,
         'VAR': <TokenType.VAR: 'VAR'>,
         'PROCEDURE': <TokenType.PROCEDURE: 'PROCEDURE'>,
         'BEGIN': <TokenType.BEGIN: 'BEGIN'>,
         'END': <TokenType.END: 'END'>}
    """
    # enumerations support iteration, in definition order
    tt_list = list(TokenType)
    start_index = tt_list.index(TokenType.INT)
    end_index = tt_list.index(TokenType.RETURN)
    reserved_keywords = {
        token_type.value: token_type
        for token_type in tt_list[start_index:end_index + 1]
    }
    return reserved_keywords


RESERVED_KEYWORDS = _build_reserved_keywords()


class Lexer:
    def __init__(self, text):
        # client string input, e.g. "4 + 2 * 3 - 6 / 2"
        self.text = text
        # self.pos is an index into self.text
        self.pos = 0
        self.current_char = self.text[self.pos]
        # token line number and column number
        self.lineno = 1
        self.column = 1

    def error(self):
        s = "Lexer error on '{lexeme}' line: {lineno} column: {column}".format(
            lexeme=self.current_char,
            lineno=self.lineno,
            column=self.column,
        )
        raise LexerError(message=s)

    def advance(self):
        """Advance the `pos` pointer and set the `current_char` variable."""
        if self.current_char == '\n':
            self.lineno += 1
            self.column = 0

        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]
            self.column += 1

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def number(self):
        """Return a (multidigit) integer or float consumed from the input."""

        # Create a new token with current line and column number
        token = Token(type=None, value=None, lineno=self.lineno, column=self.column)

        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        if self.current_char == '.':
            result += self.current_char
            self.advance()

            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self.advance()

            token.type = TokenType.REAL_CONST
            token.value = float(result)
        else:
            token.type = TokenType.INTEGER_CONST
            token.value = int(result)

        return token

    def _id(self):
        """Handle identifiers and reserved keywords"""

        # Create a new token with current line and column number
        token = Token(type=None, value=None, lineno=self.lineno, column=self.column)

        value = ''
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            value += self.current_char
            self.advance()

        token_type = RESERVED_KEYWORDS.get(value.upper())
        if token_type is None:
            token.type = TokenType.ID
            token.value = value
        else:
            # reserved keyword
            token.type = token_type
            token.value = value.upper()

        return token

    def lexer(self):
        tokens = []
        while True:
            tok = self.get_next_token()
            tokens.append(tok)
            if tok.type == TokenType.EOF:
                break
        return tokens

    def get_escape_char(self, c):
        if c == 'a': return '\a'
        if c == 'b': return '\b'
        if c == 't': return '\t'
        if c == 'n': return '\n'
        if c == 'v': return '\v'
        if c == 'f': return '\f'
        if c == 'r': return '\r'
        if c == 'e': return 27
        if c == '0': return 0
        return c

    def read_string_literal(self):
        value = ''
        while self.current_char is not None:
            if self.current_char == '"':
                self.advance()
                break
            if self.current_char == '\\':
                self.advance()
                try:
                    value += self.get_escape_char(self.current_char)
                except:
                    value += chr(self.get_escape_char(self.current_char))
                self.advance()
            else:
                value += self.current_char
                self.advance()
        value += '\0'
        tok = Token(
            type=TokenType.STR,
            value=value,
            lineno=self.lineno,
            column=self.column)
        return tok

    def get_next_token(self):
        """Lexical analyzer (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
        while self.current_char is not None:
            if self.current_char == '/':
                if self.peek() == '/':
                    self.advance()
                    self.advance()
                    while self.current_char != '\n':
                        self.advance()
                    continue
                if self.peek() == '*':
                    if '*/' not in self.text[self.pos:]:
                        raise Exception('注释没有闭合')
                    idx = self.text.index('*/', self.pos)
                    while self.pos != idx + 2:
                        self.advance()
                    continue

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isalpha():
                return self._id()

            if self.current_char.isdigit():
                return self.number()

            if self.current_char == '>' and self.peek() == '=':
                token = Token(
                    type=TokenType.GE,
                    value='>=',
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                self.advance()
                return token

            if self.current_char == '>':
                token = Token(
                    type=TokenType.GT,
                    value='>',
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                return token

            if self.current_char == '<' and self.peek() == '=':
                token = Token(
                    type=TokenType.LE,
                    value='<=',
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                self.advance()
                return token

            if self.current_char == '<':
                token = Token(
                    type=TokenType.LT,
                    value='<',
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                return token

            if self.current_char == '=' and self.peek() == '=':
                token = Token(
                    type=TokenType.EQ,
                    value='==',
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                self.advance()
                return token

            if self.current_char == '=':
                token = Token(
                    type=TokenType.ASSIGN,
                    value='=',
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                return token

            if self.current_char == '!' and self.peek() == '=':
                token = Token(
                    type=TokenType.NE,
                    value=TokenType.NE.value,  # ':='
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                self.advance()
                return token

            if self.current_char == '"':
                self.advance()
                token = self.read_string_literal()

                return token

            # single-character token
            try:
                # get enum member by value, e.g.
                # TokenType(';') --> TokenType.SEMI
                token_type = TokenType(self.current_char)
            except ValueError:
                # no enum member with value equal to self.current_char
                self.error()
            else:
                # create a token with a single-character lexeme as its value
                token = Token(
                    type=token_type,
                    value=token_type.value,  # e.g. ';', '.', etc
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                return token

        # EOF (end-of-file) token indicates that there is no more
        # input left for lexical analysis
        return Token(type=TokenType.EOF, value=None)

# AST node
class NodeKind(Enum):
    ND_ADD       = auto() # num + num
    ND_PTR_ADD   = auto() # ptr + num or num + ptr
    ND_SUB       = auto() # num - num
    ND_PTR_SUB   = auto() # ptr - num
    ND_PTR_DIFF  = auto() # ptr - ptr
    ND_MUL       = auto() # *
    ND_DIV       = auto() # /
    ND_EQ        = auto() # ==
    ND_NE        = auto() # !=
    ND_LT        = auto() # <
    ND_LE        = auto() # <=
    ND_ASSIGN    = auto() # =
    ND_MEMBER    = auto() # . (struct member access)
    ND_ADDR      = auto() # unary &
    ND_DEREF     = auto() # unary *
    ND_RETURN    = auto() # "return"
    ND_IF        = auto() # "if"
    ND_WHILE     = auto() # "while"
    ND_FOR       = auto() # "for"
    ND_BLOCK     = auto() # { ... }
    ND_FUNCALL   = auto() # Function call
    ND_EXPR_STMT = auto() # Expression statement
    ND_STMT_EXPR = auto() # Statement expression
    ND_VAR       = auto() # Variable
    ND_NUM       = auto() # Integer
    ND_NULL      = auto() # Empty statement

class Node:
    def __init__(self):
        self.kind     = None    # Node kind
        self.tok      = None    # Representative token
        self.ty       = None    # Type, e.g. int or pointer to int

        self.lhs      = None    # Left-hand side
        self.rhs      = None    # Right-hand side

        # "if, "while" or "for" statement
        self.cond     = None
        self.then     = None
        self.els      = None
        self.init     = None
        self.inc      = None

        # Block or statement expression
        self.body     = None

        # Struct member access
        self.member   = None

        # Function call
        self.funcname = None
        self.args     = None

        self.var      = None      # Used if kind == ND_VAR
        self.val      = None      # Used if kind == ND_NUM

class AST:
    pass

class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right
        self.ty = None

class UnaryOp(AST):
    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr
        self.ty = None

class Assign(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right
        self.ty = None

class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value
        self.ty = None

class ExprStmt(AST):
    def __init__(self, expr):
        self.token = Token(TokenType.EXPRSTMT, None)
        self.expr = expr
        self.ty = None

class StmtExpr(AST):
    def __init__(self):
        self.token = Token(TokenType.STMTEXPR, None)
        self.body = []
        self.ty = None

class Var(AST):
    def __init__(self):
        self.name = None
        self.offset = 0
        self.token = Token(TokenType.VAR, None)
        self.ty = None
        self.is_local = None
        self.contents = None

class Function(AST):
    def __init__(self):
        self.name = None
        self.nodes = []
        self.params = []
        self.locals = []
        self.stack_size = 0
        self.ty = None

class IfStmt(AST):
    def __init__(self):
        self.token = Token(TokenType.IFSTMT, None)
        self.cond = None
        self.then = None
        self.els = None
        self.ty = None

class WhileStmt(AST):
    def __init__(self):
        self.token = Token(TokenType.WHILESTMT, None)
        self.cond = None
        self.then = None
        self.ty = None

class ForStmt(AST):
    def __init__(self):
        self.token = Token(TokenType.FORSTMT, None)
        self.init = None
        self.cond = None
        self.then = None
        self.inc  = None
        self.ty = None

class Block(AST):
    def __init__(self):
        self.token = Token(TokenType.BLOCK, None)
        self.body = []
        self.ty = None

class FunCall(AST):
    def __init__(self):
        self.token = Token(TokenType.FUNCALL, None)
        self.funcname = None
        self.args     = []
        self.ty = None

class Addr(AST):
    def __init__(self):
        self.token = Token(TokenType.ADDR, None)
        self.expr = None
        self.ty = None

class Deref(AST):
    def __init__(self):
        self.token = Token(TokenType.DEREF, None)
        self.expr = None
        self.ty = None

class PtrAdd(AST):
    def __init__(self):
        self.token = self.op = Token(TokenType.PTRADD, None)
        self.ty = None
        self.left = None
        self.right = None

class PtrSub(AST):
    def __init__(self):
        self.token = self.op = Token(TokenType.PTRSUB, None)
        self.ty = None
        self.left = None
        self.right = None

class PtrDiff(AST):
    def __init__(self):
        self.token = self.op = Token(TokenType.PTRDIFF, None)
        self.ty = None
        self.left = None
        self.right = None

class NullAst(AST):
    def __init__(self):
        self.token = Token(TokenType.NULL, None)
        self.ty = None

class Program(AST):
    def __init__(self):
        self.globals = []
        self.fns = []

class Member:
    def __init__(self):
        self.ty = None
        self.name = None
        self.offset = 0

class MemberAst(AST):
    def __init__(self):
        self.expr = None
        self.member = None
        self.ty = None
        self.token = Token(TokenType.MEMBER, None)

class Type:
    def __init__(self):
        self.kind = None
        self.base = None    # 指针或者数组
        self.size = 0       # sizeof()的值
        self.array_len = 0
        self.members = []   # struct

class IntType(Type):
    def __init__(self):
        self.kind = TokenType.TYINT
        self.base = None
        self.size = 8

class CharType(Type):
    def __init__(self):
        self.kind = TokenType.TYCHAR
        self.base = None
        self.size = 1

class Parser:
    """
    program    = stmt*
    stmt       = expr ";"
    expr       = assign
    assign     = equality ("=" assign)?
    equality   = relational ("==" relational | "!=" relational)*
    relational = add ("<" add | "<=" add | ">" add | ">=" add)*
    add        = mul ("+" mul | "-" mul)*
    mul        = unary ("*" unary | "/" unary)*
    unary      = ("+" | "-")? primary
    primary    = num | ident | "(" expr ")"
    """
    def __init__(self, tokens):
        # set current token to the first token taken from the input
        self.current_token_index = 0
        self.locals = []
        self.globals = []
        self.scope = []
        self.labelseq = 1
        self.funcname = None
        self.tokens = tokens
        self.current_token = self.tokens[self.current_token_index]
        self.cnt = 0

    def find_var(self, token):
        for v in self.scope:
            if v.name == token.value:
                return v
        return None

    def new_node(self, kind, tok):
        node = Node()
        node.kind = kind
        node.tok = tok
        return node

    def new_binary(self, kind, lhs, rhs, tok):
        node = self.new_node(kind, tok)
        node.lhs = lhs
        node.rhs = rhs
        return node

    def new_unary(self, kind, expr, tok):
        node = self.new_node(kind, tok)
        node.lhs = expr
        return node

    def new_num(self, val, tok):
        node = self.new_node(NodeKind.ND_NUM, tok)
        node.val = val
        return node

    def new_var_node(self, var, tok):
        node = self.new_node(NodeKind.ND_VAR, tok)
        node.var = var
        return node

    def get_next_token(self):
        self.current_token_index += 1
        return self.tokens[self.current_token_index]

    def error(self, error_code, token):
        raise ParserError(
            error_code=error_code,
            token=token,
            message=f'{error_code.value} -> {token}',
        )

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.get_next_token()
        else:
            self.error(
                error_code=ErrorCode.UNEXPECTED_TOKEN,
                token=self.current_token,
            )

    def consume(self, token_type):
        try:
            self.eat(token_type)
            return True
        except:
            return False

    def new_gvar(self, name, ty):
        var = self.new_var(name, ty, False)
        self.globals.insert(0, var)
        return var

    def new_var(self, name, ty, is_local):
        var = Var()
        var.name = name
        var.ty = ty
        var.is_local = is_local

        self.scope.insert(0, var)
        return var

    def new_lvar(self, name, ty):
        var = self.new_var(name, ty, True)
        self.locals.insert(0, var)

        return var

    def is_function(self):
        idx = self.current_token_index
        self.basetype()
        if self.current_token.type != TokenType.ID:
            return False
        self.eat(TokenType.ID)
        if self.current_token.type != TokenType.LPAREN:
            self.current_token_index = idx
            self.current_token = self.tokens[self.current_token_index]
            return False
        self.eat(TokenType.LPAREN)
        self.current_token_index = idx
        self.current_token = self.tokens[self.current_token_index]
        return True

    # global-var = basetype ident ("[" num "]")* ";"
    def global_var(self):
        ty = self.basetype()
        name = self.current_token.value
        self.eat(TokenType.ID)
        ty = self.read_type_suffix(ty)
        self.eat(TokenType.SEMI)
        self.new_gvar(name, ty)

    # stmt-expr = "(" "{" stmt stmt* "}" ")"
    # Statement expression is a GNU C extension.
    def stmt_expr(self, tok):
        node = StmtExpr()
        node.body = [self.stmt()]
        while True:
            if self.consume(TokenType.RBRACE):
                break
            else:
                node.body.append(self.stmt())
        self.eat(TokenType.RPAREN)
        node.body[-1] = node.body[-1].expr
        return node

    # program = (global-var | function)*
    def program(self):
        function_list = []
        self.globals = []
        while self.current_token.type != TokenType.EOF:
            if self.is_function():
                function_list.append(self.function())
            else:
                self.global_var()

        prog = Program()
        prog.fns = function_list
        prog.globals = self.globals
        return prog

    def function(self):
        self.locals = []
        fn = Function()
        self.basetype()
        if self.current_token.type == TokenType.ID:
            fn.name = self.current_token.value
            self.eat(TokenType.ID)
        self.eat(TokenType.LPAREN)
        sc = self.scope[:]
        fn.params = self.read_func_params()
        self.eat(TokenType.LBRACE)

        stmts = []

        while True:
            try:
                self.eat(TokenType.RBRACE)
                break
            except:
                stmts.append(self.stmt())
        self.scope = sc[:]
        fn.nodes = stmts
        fn.locals = self.locals
        return fn

    # declaration = basetype ident ("[" num "]")* ("=" expr) ";"
    def declaration(self):
        token = self.current_token
        ty = self.basetype()
        name = self.current_token.value
        self.eat(TokenType.ID)
        ty = self.read_type_suffix(ty)
        var = self.new_lvar(name, ty)

        if self.current_token.type == TokenType.SEMI:
            self.eat(TokenType.SEMI)
            return NullAst()

        self.eat(TokenType.ASSIGN)
        left = var
        right = self.expr()
        self.eat(TokenType.SEMI)
        node = Assign(left=left, op=Token(TokenType.ASSIGN, value='='), right=right)
        return ExprStmt(expr=node)

    def stmt(self):
        node = self.stmt2()
        self.add_type(node)
        return node

    def stmt2(self):
        token = self.current_token
        if token.type == TokenType.RETURN:
            self.eat(TokenType.RETURN)
            node = UnaryOp(op=token, expr=self.expr())
            self.eat(TokenType.SEMI)
            return node

        if self.consume(TokenType.IF):
            node = IfStmt()
            # if self.current_token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node.cond = self.expr()
            # if self.current_token.type == TokenType.RPAREN:
            self.eat(TokenType.RPAREN)
            node.then = self.stmt()
            if self.consume(TokenType.ELSE):
                node.els = self.stmt()
            return node

        if self.consume(TokenType.WHILE):
            node = WhileStmt()
            self.eat(TokenType.LPAREN)
            node.cond = self.expr()
            self.eat(TokenType.RPAREN)
            node.then = self.stmt()
            return node

        if self.consume(TokenType.FOR):
            node = ForStmt()
            self.eat(TokenType.LPAREN)
            while not self.consume(TokenType.SEMI):
                node.init = ExprStmt(self.expr())
            while not self.consume(TokenType.SEMI):
                node.cond = self.expr()
            while not self.consume(TokenType.RPAREN):
                node.inc = ExprStmt(self.expr())
            node.then = self.stmt()
            return node

        if self.consume(TokenType.LBRACE):
            stmt_list = []

            sc = self.scope[:]
            while True:
                try:
                    self.eat(TokenType.RBRACE)
                    break
                except:
                    stmt_list.append(self.stmt())
            self.scope = sc[:]

            node = Block()
            node.body = stmt_list
            return node

        if self.is_typename():
            return self.declaration()

        node = ExprStmt(self.expr())
        self.eat(TokenType.SEMI)
        return node

    def is_typename(self):
        return self.current_token.type == TokenType.INT or \
        self.current_token.type == TokenType.CHAR or \
        self.current_token.type == TokenType.STRUCT

    def expr(self):
        return self.assign()

    def assign(self):
        node = self.equality()

        token = self.current_token
        if self.consume(TokenType.ASSIGN):
            node = Assign(left=node, op=token, right=self.assign())
        return node

    def equality(self):
        node = self.relational()

        while self.current_token.type in (TokenType.EQ, TokenType.NE):
            token = self.current_token
            self.eat(token.type)
            
            node = BinOp(left=node, op=token, right=self.relational())

        return node

    def relational(self):
        node = self.add()

        while self.current_token.type in (TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE):
            token = self.current_token
            self.eat(token.type)
            node = BinOp(left=node, op=token, right=self.add())

        return node

    def add(self):
        node = self.mul()

        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            token = self.current_token
            if self.consume(TokenType.PLUS):
                node = self.new_add(node, self.mul(), token)
            elif self.consume(TokenType.MINUS):
                node = self.new_sub(node, self.mul(), token)

        return node

    def mul(self):
        node = self.unary()

        while self.current_token.type in (TokenType.MUL, TokenType.DIV):
            token = self.current_token
            self.eat(token.type)
            node = BinOp(left=node, op=token, right=self.unary())

        return node

    # postfix = primary ("[" expr "]" | "." ident)*
    def postfix(self):
        node = self.primary()

        while True:
            if self.consume(TokenType.LBRACKET):
                # x[y] 是 *(x+y) 的语法糖
                tok = self.current_token
                exp = self.new_add(node, self.expr(), tok)
                self.eat(TokenType.RBRACKET)
                node = Deref()
                node.expr = exp
                continue

            if self.consume(TokenType.DOT):
                node = self.struct_ref(node)
                continue

            return node

    # primary = "(" "{" stmt-expr-tail
    #         | "(" expr ")"
    #         | "sizeof" unary
    #         | ident func-args?
    #         | str
    #         | num
    def primary(self):
        token = self.current_token
        if self.consume(TokenType.LPAREN):
            if self.consume(TokenType.LBRACE):
                return self.stmt_expr(token)
            node = self.expr()
            self.eat(TokenType.RPAREN)
            return node

        if self.consume(TokenType.SIZEOF):
            node = self.unary()
            self.add_type(node)
            return Num(token=Token(type=TokenType.INTEGER_CONST, value=node.ty.size))

        if self.consume(TokenType.ID):

            # 函数调用
            if self.consume(TokenType.LPAREN):
                node = FunCall()
                node.funcname = token.value
                node.args = self.func_args()
                return node

            # 查找变量
            var = self.find_var(token)
            if var is None:
                raise Exception("没有找到变量")
            return var

        if token.type == TokenType.STR:
            self.current_token = self.get_next_token()

            ty = self.array_of(CharType(), len(token.value))
            var = self.new_gvar(self.new_label(), ty)
            var.contents = token.value
            return var

        self.current_token = self.get_next_token()

        return Num(token)

    def new_label(self):
        s = ".L.data.%d" % self.cnt
        self.cnt += 1
        return s

    # unary = ("+" | "-" | "*" | "&")? unary
    #       | postfix
    def unary(self):
        token = self.current_token
        if self.consume(TokenType.PLUS):
            return self.primary()
        if self.consume(TokenType.MINUS):
            zero = Num(Token(type=TokenType.INTEGER_CONST, value=0))
            return BinOp(left=zero, op=token, right=self.unary())
        if self.consume(TokenType.ADDR):
            node = Addr()
            node.expr = self.unary()
            return node
        if self.consume(TokenType.MUL):
            node = Deref()
            node.expr = self.unary()
            return node
        return self.postfix()

    def func_args(self):
        if self.consume(TokenType.RPAREN):
            return []
        args_list = [self.assign()]
        while True:
            if self.consume(TokenType.COMMA):
                args_list.append(self.assign())
            else:
                break
        self.eat(TokenType.RPAREN)
        return args_list

    def read_type_suffix(self, base):
        if self.current_token.type != TokenType.LBRACKET:
            return base
        self.eat(TokenType.LBRACKET)
        sz = Num(self.current_token)
        self.current_token = self.get_next_token()
        self.eat(TokenType.RBRACKET)
        base = self.read_type_suffix(base)
        return self.array_of(base, sz.value)

    def read_func_param(self):
        ty = self.basetype()
        name = self.current_token.value
        self.eat(TokenType.ID)
        ty = self.read_type_suffix(ty)
        var = self.new_lvar(name, ty)
        return var

    def read_func_params(self):
        if self.consume(TokenType.RPAREN):
            return []
        
        var_list = []
        v = self.read_func_param()
        var_list.append(v)

        while True:
            try:
                self.eat(TokenType.RPAREN)
                break
            except:
                self.eat(TokenType.COMMA)
                v = self.read_func_param()
                var_list.append(v)
        
        return var_list

    def new_add(self, left, right, tok):
        self.add_type(left)
        self.add_type(right)
        if self.is_integer(left.ty) and self.is_integer(right.ty):
            return BinOp(left=left, op=Token(TokenType.PLUS, None), right=right)
        if left.ty.base and self.is_integer(right.ty):
            node = PtrAdd()
            node.left = left
            node.right = right
            return node
        if self.is_integer(left.ty) and right.ty.base:
            node = PtrAdd()
            node.left = right
            node.right = left
            return node

    def new_sub(self, left, right, tok):
        self.add_type(left)
        self.add_type(right)

        if self.is_integer(left.ty) and self.is_integer(right.ty):
            return BinOp(left=left, op=tok, right=right)
        if left.ty.base and self.is_integer(right.ty):
            node = PtrSub()
            node.left = left
            node.right = right
            return node
        if left.ty.base and right.ty.base:
            node = PtrDiff()
            node.left = left
            node.right = right
            return node

    def add_type(self, node):
        if node is None or node.ty is not None:
            return

        if hasattr(node, 'left'):
            self.add_type(node.left)
        if hasattr(node, 'right'):
            self.add_type(node.right)
        if hasattr(node, 'cond'):
            self.add_type(node.cond)
        if hasattr(node, 'then'):
            self.add_type(node.then)
        if hasattr(node, 'els'):
            self.add_type(node.els)
        if hasattr(node, 'init'):
            self.add_type(node.init)
        if hasattr(node, 'inc'):
            self.add_type(node.inc)
        if hasattr(node, 'expr'):
            self.add_type(node.expr)

        if hasattr(node, 'body'):
            for n in node.body:
                self.add_type(n)
        if hasattr(node, 'args'):
            for n in node.args:
                self.add_type(n)

        if node.token.type == TokenType.PLUS or    \
           node.token.type == TokenType.MINUS or   \
           node.token.type == TokenType.PTRDIFF or \
           node.token.type == TokenType.MUL or     \
           node.token.type == TokenType.DIV or     \
           node.token.type == TokenType.EQ or      \
           node.token.type == TokenType.NE or      \
           node.token.type == TokenType.LT or      \
           node.token.type == TokenType.LE or      \
           node.token.type == TokenType.VAR or     \
           node.token.type == TokenType.FUNCALL or \
           node.token.type == TokenType.INTEGER_CONST:
            node.ty = IntType()
            return

        if node.token.type == TokenType.PTRADD or \
           node.token.type == TokenType.PTRSUB or \
           node.token.type == TokenType.ASSIGN:
            node.ty = node.left.ty
            return

        if node.token.type == TokenType.ADDR:
            if node.expr.ty.kind == TokenType.TYARRAY:
                node.ty = self.pointer_to(node.expr.ty.base)
            else:
                node.ty = self.pointer_to(node.expr.ty)
            return

        if node.token.type == TokenType.DEREF:
            if node.expr.ty.base is not None:
                node.ty = node.expr.ty.base
            else:
                node.ty = IntType()
            return

        if node.token.type == TokenType.MEMBER:
            node.ty = node.member.ty
            return

        if node.token.type == TokenType.STMTEXPR:
            node.ty = node.body[-1].ty
            return

    def is_integer(self, ty):
        return ty.kind == TokenType.TYINT or ty.kind == TokenType.TYCHAR

    def pointer_to(self, base):
        ty = Type()
        ty.kind = TokenType.TYPTR
        ty.size = 8
        ty.base = base
        return ty

    def array_of(self, base, len):
        ty = Type()
        ty.kind = TokenType.TYARRAY
        ty.size = base.size * len
        ty.base = base
        ty.array_len = len
        return ty

    # basetype = ("char" | "int" | struct-decl) "*"*
    def basetype(self):
        if self.is_typename() is False:
            raise Exception('这里应该是一个类型名')
        if self.current_token.type == TokenType.CHAR:
            self.eat(TokenType.CHAR)
            ty = CharType()
        elif self.current_token.type == TokenType.INT:
            self.eat(TokenType.INT)
            ty = IntType()
        else:
            ty = self.struct_decl()
        while True:
            try:
                self.eat(TokenType.MUL)
                ty = self.pointer_to(ty)
            except:
                break

        return ty

    # struct-decl = "struct" "{" struct-member "}"
    def struct_decl(self):
        self.eat(TokenType.STRUCT)
        self.eat(TokenType.LBRACE)

        members = []
        while True:
            try:
                self.eat(TokenType.RBRACE)
                break
            except:
                members.append(self.struct_member())

        ty = Type()
        ty.kind = TokenType.TYSTRUCT
        ty.members = members
        
        offset = 0
        for m in ty.members:
            m.offset = offset
            offset += m.ty.size
        ty.size = offset

        return ty

    # struct-member = basetype ident ("[" num "]")* ";"
    def struct_member(self):
        mem = Member()
        mem.ty = self.basetype()
        mem.name = self.current_token.value
        self.eat(TokenType.ID)
        mem.ty = self.read_type_suffix(mem.ty)
        self.eat(TokenType.SEMI)
        return mem

    def find_member(self, ty, name):
        for mem in ty.members:
            if mem.name == name:
                return mem
        return None

    def struct_ref(self, expr):
        self.add_type(expr)
        if expr.ty.kind != TokenType.TYSTRUCT:
            raise Exception('不是一个结构体')

        mem = self.find_member(expr.ty, self.current_token.value)
        self.eat(TokenType.ID)
        if mem is None:
            raise Exception('没有这个成员')

        node = MemberAst()
        node.expr = expr
        node.member = mem
        return node

    def gen_addr(self, node):
        if node.token.type == TokenType.VAR:
            if node.is_local:
                print("  lea rax, [rbp-%d]" % node.offset)
                print("  push rax")
            else:
                print("  push offset %s" % node.name)
            return
        if node.token.type == TokenType.DEREF:
            self.code_gen(node.expr)
            return
        if node.token.type == TokenType.MEMBER:
            self.gen_addr(node.expr)
            print("  pop rax")
            print("  add rax, %d" % node.member.offset)
            print("  push rax")
            return
        
        self.error(
            error_code=ErrorCode.UNEXPECTED_TOKEN,
            token=self.current_token,
        )

    def gen_lvar(self, node):
        if node.ty.kind == TokenType.TYARRAY:
            raise Exception(node.token + '不是左值')
        self.gen_addr(node)

    def load(self, ty):
        print("  pop rax")
        if ty.size == 1:
            print("  movsx rax, byte ptr [rax]")
        else:
            print("  mov rax, [rax]")
        print("  push rax")

    def store(self, ty):
        print("  pop rdi")
        print("  pop rax")
        if ty.size == 1:
            print("  mov [rax], dil")
        else:
            print("  mov [rax], rdi")
        print("  push rdi")

    def load_arg(self, var, idx):
        sz = var.ty.size
        if sz == 1:
            print("  mov [rbp-%d], %s" % (var.offset, argreg1[idx]))
        else:
            assert sz == 8
            print("  mov [rbp-%d], %s" % (var.offset, argreg8[idx]))

    def emit_data(self, prog):
        print(".data")
        for var in prog.globals:
            print("%s:" % var.name)
            if not var.contents:
                print("  .zero %d" % var.ty.size)
                continue
            for c in var.contents:
                print("  .byte %d" % ord(c))

    def align_to(self, n, align):
        return (n + align - 1) & ~(align - 1)

    def code_gen(self, node):
        if node.token.type == TokenType.NULL:
            return
        if node.token.type == TokenType.INTEGER_CONST:
            print("  push %s" % node.value)
            return
        if node.token.type == TokenType.EXPRSTMT:
            self.code_gen(node.expr)
            print("  add rsp, 8")
            return
        if node.token.type == TokenType.VAR or node.token.type == TokenType.MEMBER:
            self.gen_addr(node)
            if node.ty.kind != TokenType.TYARRAY:
                self.load(node.ty)
            return
        if node.token.type == TokenType.ASSIGN:
            self.gen_lvar(node.left)
            self.code_gen(node.right)
            self.store(node.ty)
            return
        if node.token.type == TokenType.ADDR:
            self.gen_addr(node.expr)
            return
        if node.token.type == TokenType.DEREF:
            self.code_gen(node.expr)
            if node.ty.kind != TokenType.TYARRAY:
                self.load(node.ty)
            return
        if node.token.type == TokenType.IFSTMT:
            seq = self.labelseq
            self.labelseq += 1
            if node.els is not None:
                self.code_gen(node.cond)
                print("  pop rax")
                print("  cmp rax, 0")
                print("  je  .L.else.%d" % seq)
                self.code_gen(node.then)
                print("  jmp .L.end.%d" % seq)
                print(".L.else.%d:" % seq)
                self.code_gen(node.els)
                print(".L.end.%d:" % seq)
            else:
                self.code_gen(node.cond)
                print("  pop rax")
                print("  cmp rax, 0")
                print("  je  .L.end.%d" % seq)
                self.code_gen(node.then)
                print(".L.end.%d:" % seq)
            return
        if node.token.type == TokenType.WHILESTMT:
            seq = self.labelseq
            self.labelseq += 1
            print(".L.begin.%d:" % seq)
            self.code_gen(node.cond)
            print("  pop rax")
            print("  cmp rax, 0")
            print("  je  .L.end.%d" % seq)
            self.code_gen(node.then)
            print("  jmp .L.begin.%d" % seq)
            print(".L.end.%d:" % seq)
            return
        if node.token.type == TokenType.FORSTMT:
            seq = self.labelseq
            self.labelseq += 1
            if node.init is not None:
                self.code_gen(node.init)
            print(".L.begin.%d:" % seq)
            if node.cond is not None:
                self.code_gen(node.cond)
                print("  pop rax")
                print("  cmp rax, 0")
                print("  je  .L.end.%d" % seq)
            self.code_gen(node.then)
            if node.inc is not None:
                self.code_gen(node.inc)
            print("  jmp .L.begin.%d" % seq)
            print(".L.end.%d:" % seq)
            return
        if node.token.type == TokenType.BLOCK or node.token.type == TokenType.STMTEXPR:
            for n in node.body:
                self.code_gen(n)
            return
        if node.token.type == TokenType.FUNCALL:
            for arg in node.args:
                self.code_gen(arg)
            for i in range(len(node.args)-1, -1, -1):
                print("  pop %s" % argreg8[i])
            # 函数调用前，必须将 RSP 对齐到 16 字节的边界
            # 这是 ABI 的要求
            seq = self.labelseq
            self.labelseq += 1
            print("  mov rax, rsp")
            print("  and rax, 15")
            print("  jnz .L.call.%d" % seq)
            print("  mov rax, 0")
            print("  call %s" % node.funcname)
            print("  jmp .L.end.%d" % seq)
            print(".L.call.%d:" % seq)
            print("  sub rsp, 8")
            print("  mov rax, 0")
            print("  call %s" % node.funcname)
            print("  add rsp, 8")
            print(".L.end.%d:" % seq)
            print("  push rax")
            return
        if node.token.type == TokenType.RETURN:
            self.code_gen(node.expr)
            print("  pop rax")
            print("  jmp .L.return.%s" % self.funcname)
            return

        self.code_gen(node.left)
        self.code_gen(node.right)

        print("  pop rdi")
        print("  pop rax")

        if node.op.type == TokenType.PLUS:
            print("  add rax, rdi")
        if node.op.type == TokenType.PTRADD:
            print("  imul rdi, %d" % node.ty.base.size)
            print("  add rax, rdi")
        if node.op.type == TokenType.MINUS:
            print("  sub rax, rdi")
        if node.op.type == TokenType.PTRSUB:
            print("  imul rdi, %d" % node.ty.base.size)
            print("  sub rax, rdi")
        if node.op.type == TokenType.PTRDIFF:
            print("  sub rax, rdi")
            print("  cqo")
            print("  mov rdi, %d" % node.left.ty.base.size)
            print("  idiv rdi")
        if node.op.type == TokenType.MUL:
            print("  imul rax, rdi")
        if node.op.type == TokenType.DIV:
            print("  cqo")
            print("  idiv rdi")
        if node.op.type == TokenType.EQ:
            print("  cmp rax, rdi")
            print("  sete al")
            print("  movzb rax, al")
        if node.op.type == TokenType.NE:
            print("  cmp rax, rdi")
            print("  setne al")
            print("  movzb rax, al")
        if node.op.type == TokenType.LT:
            print("  cmp rax, rdi")
            print("  setl al")
            print("  movzb rax, al")
        if node.op.type == TokenType.LE:
            print("  cmp rax, rdi")
            print("  setle al")
            print("  movzb rax, al")
        if node.op.type == TokenType.GT:
            print("  cmp rax, rdi")
            print("  setg al")
            print("  movzb rdi, al")
        if node.op.type == TokenType.GE:
            print("  cmp rax, rdi")
            print("  setge al")
            print("  movzb rdi, al")

        print("  push rax")

    def parse(self):
        nodes = self.program()
        if self.current_token.type != TokenType.EOF:
            self.error(
                error_code=ErrorCode.UNEXPECTED_TOKEN,
                token=self.current_token,
            )

        return nodes


def main():
    parser = argparse.ArgumentParser(
        description='tutucc - simple c compiler'
    )
    parser.add_argument('inputfile', help='c source file')
    args = parser.parse_args()

    text = open(args.inputfile, 'r').read()
    # text = args.inputfile

    tokens = Lexer(text).lexer()
    try:
        parser = Parser(tokens)
        prog = parser.parse()
        for fn in prog.fns:
            offset = 0
            # 局部变量是顺序进入数组的，所以需要逆序弹出算偏移量。
            for v in fn.locals:
                offset += v.ty.size
                v.offset = offset
            fn.stack_size = parser.align_to(offset, 8)
        print(".intel_syntax noprefix")
        parser.emit_data(prog)
        print(".text")
        for fn in prog.fns:
            print(".global %s" % fn.name)
            print("%s:" % fn.name)
            parser.funcname = fn.name

            print("  push rbp")
            print("  mov rbp, rsp")
            print("  sub rsp, %d" % fn.stack_size)

            for i in range(len(fn.params)):
                parser.load_arg(fn.params[i], i)

            for n in fn.nodes:
                parser.code_gen(n)
            print(".L.return.%s:" % parser.funcname)
            print("  mov rsp, rbp")
            print("  pop rbp")
            print("  ret")
        sys.exit(0)
    except (LexerError, ParserError) as e:
        print(e.message)
        sys.exit(1)

if __name__ == '__main__':
    main()