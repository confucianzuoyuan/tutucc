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
    ARROW         = '->'
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
    TYPEDEF       = 'TYPEDEF'
    RETURN        = 'RETURN'
    # misc
    ID            = 'ID'
    INTEGER_CONST = 'INTEGER_CONST'
    REAL_CONST    = 'REAL_CONST'
    ASSIGN        = '='
    EOF           = 'EOF'
    ADDR          = '&'
    STR           = 'STR'

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
            if self.current_char == '-':
                if self.peek() == '>':
                    token = Token(
                        type=TokenType.ARROW,
                        value='->',
                        lineno=self.lineno,
                        column=self.column,
                    )
                    self.advance()
                    self.advance()
                    return token

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

class TypeKind(Enum):
    TY_CHAR    = auto()
    TY_INT     = auto()
    TY_PTR     = auto()
    TY_ARRAY   = auto()
    TY_STRUCT  = auto()

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

        self.next     = None

class Var:
    def __init__(self):
        self.name = None
        self.offset = 0
        self.ty = None
        self.is_local = None
        self.contents = None

class Function:
    def __init__(self):
        self.name = None
        self.node = None
        self.params = None
        self.locals = None
        self.stack_size = 0
        self.ty = None
        self.next = None

class Program:
    def __init__(self):
        self.globals = None
        self.fns = None

class Member:
    def __init__(self):
        self.ty = None
        self.name = None
        self.offset = 0
        self.next = None

class Type:
    def __init__(self):
        self.kind = None
        self.base = None    # 指针或者数组
        self.size = 0       # sizeof()的值
        self.align = 0
        self.array_len = 0
        self.members = None   # struct

# C has two block scopes; one is for variables/typedefs and
# the other is for struct tags.
class TagScope:
    def __init__(self):
        self.next = None
        self.name = None
        self.ty   = None

class VarScope:
    def __init__(self):
        self.var = None
        self.next = None
        self.name = None
        self.type_def = None

class Scope:
    def __init__(self):
        self.var_scope = None
        self.tag_scope = None

class VarList:
    def __init__(self):
        self.var  = None
        self.next = None

IntType = Type()
IntType.kind = TypeKind.TY_INT
IntType.size = 8
IntType.align = 8

CharType = Type()
CharType.kind = TypeKind.TY_CHAR
CharType.size = 1
CharType.align = 1

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
        self.locals = None
        self.globals = None
        self.var_scope = None
        self.tag_scope = None
        self.labelseq = 1
        self.funcname = None
        self.tokens = tokens
        self.current_token = self.tokens[self.current_token_index]
        self.cnt = 0

    def enter_scope(self):
        sc = Scope()
        sc.var_scope = self.var_scope
        sc.tag_scope = self.tag_scope
        return sc

    def leave_scope(self, sc):
        self.var_scope = sc.var_scope
        self.tag_scope = sc.tag_scope

    def push_scope(self, name):
        sc = VarScope()
        sc.name = name
        sc.next = self.var_scope
        self.var_scope = sc
        return sc

    def push_tag_scope(self, token, ty):
        sc = TagScope()
        sc.next = self.tag_scope
        sc.name = token.value
        sc.ty   = ty
        self.tag_scope = sc

    def find_tag(self, token):
        sc = self.tag_scope
        while sc is not None:
            if sc.name == token.value:
                return sc
            sc = sc.next
        return None

    def find_var(self, token):
        sc = self.var_scope
        while sc is not None:
            if sc.name == token.value:
                return sc
            sc = sc.next
        return None

    def find_typedef(self, token):
        if token.type == TokenType.ID:
            sc = self.find_var(token)
            if sc:
                return sc.type_def
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
        self.push_scope(name).var = var

        vl = VarList()
        vl.var = var
        vl.next = self.globals
        self.globals = vl

        return var

    def new_var(self, name, ty, is_local):
        var = Var()
        var.name = name
        var.ty = ty
        var.is_local = is_local

        return var

    def new_lvar(self, name, ty):
        var = self.new_var(name, ty, True)
        self.push_scope(name).var = var

        vl = VarList()
        vl.var = var
        vl.next = self.locals
        self.locals = vl

        return var

    def consume_ident(self):
        token = self.current_token
        if token.type != TokenType.ID:
            return None
        self.eat(TokenType.ID)
        return token
        
    def is_function(self):
        idx = self.current_token_index
        self.basetype()
        isfunc = self.consume_ident() is not None and self.consume(TokenType.LPAREN)
        self.current_token_index = idx
        self.current_token = self.tokens[self.current_token_index]
        return isfunc

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
        sc = self.enter_scope()
        node = self.new_node(NodeKind.ND_STMT_EXPR, tok)
        node.body = self.stmt()
        cur = node.body
        while True:
            if self.consume(TokenType.RBRACE):
                break
            else:
                cur.next = self.stmt()
                cur = cur.next
        self.eat(TokenType.RPAREN)
        self.leave_scope(sc)
        # 将cur指向的节点，替换为cur.lhs指向的节点
        tmp = node.body
        while tmp.next != cur:
            tmp = tmp.next
        tmp.next = cur.lhs
        return node

    # program = (global-var | function)*
    def program(self):
        head = Function()
        cur = head
        self.globals = None
        while self.current_token.type != TokenType.EOF:
            if self.is_function():
                cur.next = self.function()
                cur = cur.next
            else:
                self.global_var()

        prog = Program()
        prog.globals = self.globals
        prog.fns = head.next
        return prog

    def function(self):
        self.locals = None
        fn = Function()
        self.basetype()
        if self.current_token.type == TokenType.ID:
            fn.name = self.current_token.value
            self.eat(TokenType.ID)
        self.eat(TokenType.LPAREN)
        sc = self.enter_scope()
        fn.params = self.read_func_params()
        self.eat(TokenType.LBRACE)

        head = Node()
        cur = head

        while True:
            try:
                self.eat(TokenType.RBRACE)
                break
            except:
                cur.next = self.stmt()
                cur = cur.next
        self.leave_scope(sc)
        fn.node = head.next
        fn.locals = self.locals
        return fn

    # declaration = basetype ident ("[" num "]")* ("=" expr) ";"
    def declaration(self):
        token = self.current_token
        ty = self.basetype()
        if self.consume(TokenType.SEMI):
            return self.new_node(NodeKind.ND_NULL, token)
        name = self.current_token.value
        self.eat(TokenType.ID)
        ty = self.read_type_suffix(ty)
        var = self.new_lvar(name, ty)

        if self.consume(TokenType.SEMI):
            return self.new_node(NodeKind.ND_NULL, token)

        self.eat(TokenType.ASSIGN)
        lhs = self.new_var_node(var, token)
        rhs = self.expr()
        self.eat(TokenType.SEMI)
        node = self.new_binary(NodeKind.ND_ASSIGN, lhs, rhs, token)
        return self.new_unary(NodeKind.ND_EXPR_STMT, node, token)

    def read_expr_stmt(self):
        token = self.current_token
        return self.new_unary(NodeKind.ND_EXPR_STMT, self.expr(), token)

    def stmt(self):
        node = self.stmt2()
        self.add_type(node)
        return node

    def stmt2(self):
        token = self.current_token
        if self.consume(TokenType.RETURN):
            node = self.new_unary(NodeKind.ND_RETURN, self.expr(), token)
            self.eat(TokenType.SEMI)
            return node

        if self.consume(TokenType.IF):
            node = self.new_node(NodeKind.ND_IF, token)
            self.eat(TokenType.LPAREN)
            node.cond = self.expr()
            self.eat(TokenType.RPAREN)
            node.then = self.stmt()
            if self.consume(TokenType.ELSE):
                node.els = self.stmt()
            return node

        if self.consume(TokenType.WHILE):
            node = self.new_node(NodeKind.ND_WHILE, token)
            self.eat(TokenType.LPAREN)
            node.cond = self.expr()
            self.eat(TokenType.RPAREN)
            node.then = self.stmt()
            return node

        if self.consume(TokenType.FOR):
            node = self.new_node(NodeKind.ND_FOR, token)
            self.eat(TokenType.LPAREN)
            while not self.consume(TokenType.SEMI):
                node.init = self.read_expr_stmt()
            while not self.consume(TokenType.SEMI):
                node.cond = self.expr()
            while not self.consume(TokenType.RPAREN):
                node.inc = self.read_expr_stmt()
            node.then = self.stmt()
            return node

        if self.consume(TokenType.LBRACE):
            head = Node()
            cur = head

            sc = self.enter_scope()
            while True:
                try:
                    self.eat(TokenType.RBRACE)
                    break
                except:
                    cur.next = self.stmt()
                    cur = cur.next
            self.leave_scope(sc)

            node = self.new_node(NodeKind.ND_BLOCK, token)
            node.body = head.next
            return node

        if self.consume(TokenType.TYPEDEF):
            ty = self.basetype()
            name = self.current_token
            self.eat(TokenType.ID)
            ty = self.read_type_suffix(ty)
            self.eat(TokenType.SEMI)
            self.push_scope(name).type_def = ty
            return self.new_node(NodeKind.ND_NULL, token)

        if self.is_typename():
            return self.declaration()

        node = self.read_expr_stmt()
        self.eat(TokenType.SEMI)
        return node

    def is_typename(self):
        return self.current_token.type == TokenType.INT or \
        self.current_token.type == TokenType.CHAR or \
        self.current_token.type == TokenType.STRUCT or \
        self.find_typedef(self.current_token)

    def expr(self):
        return self.assign()

    def assign(self):
        node = self.equality()

        token = self.current_token
        if self.consume(TokenType.ASSIGN):
            node = self.new_binary(NodeKind.ND_ASSIGN, node, self.assign(), token)
        return node

    def equality(self):
        node = self.relational()

        while True:
            token = self.current_token
            if self.consume(TokenType.EQ):
                node = self.new_binary(NodeKind.ND_EQ, node, self.relational(), token)
            elif self.consume(TokenType.NE):
                node = self.new_binary(NodeKind.ND_NE, node, self.relational(), token)
            else:
                return node

    def relational(self):
        node = self.add()

        while True:
            token = self.current_token
            if self.consume(TokenType.LT):
                node = self.new_binary(NodeKind.ND_LT, node, self.add(), token)
            elif self.consume(TokenType.LE):
                node = self.new_binary(NodeKind.ND_LE, node, self.add(), token)
            elif self.consume(TokenType.GT):
                node = self.new_binary(NodeKind.ND_LT, self.add(), node, token)
            elif self.consume(TokenType.GE):
                node = self.new_binary(NodeKind.ND_LE, self.add(), node, token)
            else:
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
        while True:
            token = self.current_token
            if self.consume(TokenType.MUL):
                node = self.new_binary(NodeKind.ND_MUL, node, self.unary(), token)
            elif self.consume(TokenType.DIV):
                node = self.new_binary(NodeKind.ND_DIV, node, self.unary(), token)
            else:
                return node

    # postfix = primary ("[" expr "]" | "." ident)*
    def postfix(self):
        node = self.primary()

        while True:
            if self.consume(TokenType.LBRACKET):
                # x[y] 是 *(x+y) 的语法糖
                token = self.current_token
                expr = self.new_add(node, self.expr(), token)
                self.eat(TokenType.RBRACKET)
                node = self.new_unary(NodeKind.ND_DEREF, expr, token)
                continue

            if self.consume(TokenType.DOT):
                node = self.struct_ref(node)
                continue

            token = self.current_token
            if self.consume(TokenType.ARROW):
                node = self.new_unary(NodeKind.ND_DEREF, node, token)
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
            return self.new_num(node.ty.size, token)

        if self.consume(TokenType.ID):

            # 函数调用
            if self.consume(TokenType.LPAREN):
                node = self.new_node(NodeKind.ND_FUNCALL, token)
                node.funcname = token.value
                node.args = self.func_args()
                return node

            # 查找变量
            sc = self.find_var(token)
            if sc and sc.var:
                return self.new_var_node(sc.var, token)
            raise Exception("没有找到变量")

        if token.type == TokenType.STR:
            self.current_token = self.get_next_token()

            ty = self.array_of(CharType, len(token.value))
            var = self.new_gvar(self.new_label(), ty)
            var.contents = token.value
            return self.new_var_node(var, token)

        return self.new_num(self.expect_number(), token)

    def expect_number(self):
        if self.current_token.type != TokenType.INTEGER_CONST:
            raise Exception('这里应该是一个数')
        val = self.current_token.value
        self.eat(TokenType.INTEGER_CONST)
        return val

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
            return self.new_binary(NodeKind.ND_SUB, self.new_num(0, token), self.unary(), token)
        if self.consume(TokenType.ADDR):
            return self.new_unary(NodeKind.ND_ADDR, self.unary(), token)
        if self.consume(TokenType.MUL):
            return self.new_unary(NodeKind.ND_DEREF, self.unary(), token)
        return self.postfix()

    def func_args(self):
        if self.consume(TokenType.RPAREN):
            return None
        head = self.assign()
        cur = head
        while True:
            if self.consume(TokenType.COMMA):
                cur.next = self.assign()
                cur = cur.next
            else:
                break
        self.eat(TokenType.RPAREN)
        return head

    def read_type_suffix(self, base):
        if self.current_token.type != TokenType.LBRACKET:
            return base
        self.eat(TokenType.LBRACKET)
        sz = self.current_token.value
        self.current_token = self.get_next_token()
        self.eat(TokenType.RBRACKET)
        base = self.read_type_suffix(base)
        return self.array_of(base, sz)

    def read_func_param(self):
        ty = self.basetype()
        name = self.current_token.value
        self.eat(TokenType.ID)
        ty = self.read_type_suffix(ty)
        vl = VarList()
        vl.var = self.new_lvar(name, ty)
        return vl

    def read_func_params(self):
        if self.consume(TokenType.RPAREN):
            return None
        
        head = self.read_func_param()
        cur = head

        while True:
            try:
                self.eat(TokenType.RPAREN)
                break
            except:
                self.eat(TokenType.COMMA)
                cur.next = self.read_func_param()
                cur = cur.next
        
        return head

    def new_add(self, lhs, rhs, token):
        self.add_type(lhs)
        self.add_type(rhs)
        if self.is_integer(lhs.ty) and self.is_integer(rhs.ty):
            return self.new_binary(NodeKind.ND_ADD, lhs, rhs, token)
        if lhs.ty.base and self.is_integer(rhs.ty):
            return self.new_binary(NodeKind.ND_PTR_ADD, lhs, rhs, token)
        if self.is_integer(lhs.ty) and rhs.ty.base:
            return self.new_binary(NodeKind.ND_PTR_ADD, rhs, lhs, token)
        raise Exception('不合法的操作符')

    def new_sub(self, lhs, rhs, tok):
        self.add_type(lhs)
        self.add_type(rhs)

        if self.is_integer(lhs.ty) and self.is_integer(rhs.ty):
            return self.new_binary(NodeKind.ND_SUB, lhs, rhs, tok)
        if lhs.ty.base and self.is_integer(rhs.ty):
            return self.new_binary(NodeKind.ND_PTR_SUB, lhs, rhs, tok)
        if lhs.ty.base and rhs.ty.base:
            return self.new_binary(NodeKind.ND_PTR_DIFF, lhs, rhs, tok)
        raise Exception('不合法的操作符')

    def add_type(self, node):
        if node is None or node.ty is not None:
            return

        self.add_type(node.lhs)
        self.add_type(node.rhs)
        self.add_type(node.cond)
        self.add_type(node.then)
        self.add_type(node.els)
        self.add_type(node.init)
        self.add_type(node.inc)

        n = node.body
        while n is not None:
            self.add_type(n)
            n = n.next
        n = node.args
        while n is not None:
            self.add_type(n)
            n = n.next

        if node.kind == NodeKind.ND_ADD or    \
           node.kind == NodeKind.ND_SUB or   \
           node.kind == NodeKind.ND_PTR_DIFF or \
           node.kind == NodeKind.ND_MUL or     \
           node.kind == NodeKind.ND_DIV or     \
           node.kind == NodeKind.ND_EQ or      \
           node.kind == NodeKind.ND_NE or      \
           node.kind == NodeKind.ND_LT or      \
           node.kind == NodeKind.ND_LE or      \
           node.kind == NodeKind.ND_FUNCALL or \
           node.kind == NodeKind.ND_NUM:
            node.ty = IntType
            return

        if node.kind == NodeKind.ND_PTR_ADD or \
           node.kind == NodeKind.ND_PTR_SUB or \
           node.kind == NodeKind.ND_ASSIGN:
            node.ty = node.lhs.ty
            return

        if node.kind == NodeKind.ND_VAR:
            node.ty = node.var.ty
            return

        if node.kind == NodeKind.ND_MEMBER:
            node.ty = node.member.ty
            return

        if node.kind == NodeKind.ND_ADDR:
            if node.lhs.ty.kind == TypeKind.TY_ARRAY:
                node.ty = self.pointer_to(node.lhs.ty.base)
            else:
                node.ty = self.pointer_to(node.lhs.ty)
            return

        if node.kind == NodeKind.ND_DEREF:
            if node.lhs.ty.base is not None:
                node.ty = node.lhs.ty.base
            else:
                node.ty = IntType
            return


        if node.kind == NodeKind.ND_STMT_EXPR:
            last = node.body
            while last.next is not None:
                last = last.next
            node.ty = last.ty
            return

    def is_integer(self, ty):
        return ty.kind == TypeKind.TY_INT or ty.kind == TypeKind.TY_CHAR

    def pointer_to(self, base):
        ty = self.new_type(TypeKind.TY_PTR, 8, 8)
        ty.base = base
        return ty

    def array_of(self, base, len):
        ty = self.new_type(TypeKind.TY_ARRAY, base.size * len, base.align)
        ty.base = base
        ty.array_len = len
        return ty

    def new_type(self, kind, size, align):
        ty = Type()
        ty.kind = kind
        ty.size = size
        ty.align = align
        return ty

    # basetype = ("char" | "int" | struct-decl | typedef-name) "*"*
    def basetype(self):
        if self.is_typename() is False:
            raise Exception('这里应该是一个类型名')
        if self.current_token.type == TokenType.CHAR:
            self.eat(TokenType.CHAR)
            ty = CharType
        elif self.current_token.type == TokenType.INT:
            self.eat(TokenType.INT)
            ty = IntType
        elif self.current_token.type == TokenType.STRUCT:
            ty = self.struct_decl()
        else:
            ty = self.find_var(self.consume_ident()).type_def
        assert ty is not None
        while True:
            try:
                self.eat(TokenType.MUL)
                ty = self.pointer_to(ty)
            except:
                break

        return ty

    # struct-decl = "struct" ident
    #             | "struct" ident? "{" struct-member "}"
    def struct_decl(self):
        self.eat(TokenType.STRUCT)
        tag = self.consume_ident()
        if tag is not None and self.current_token.type != TokenType.LBRACE:
            sc = self.find_tag(tag)
            if sc is None:
                raise Exception("未知结构体类型")
            return sc.ty
        self.eat(TokenType.LBRACE)

        head = Member()
        cur = head
        while True:
            try:
                self.eat(TokenType.RBRACE)
                break
            except:
                cur.next = self.struct_member()
                cur = cur.next

        ty = Type()
        ty.kind = TypeKind.TY_STRUCT
        ty.members = head.next
        
        offset = 0
        mem = ty.members
        while mem is not None:
            offset = self.align_to(offset, mem.ty.align)
            mem.offset = offset
            offset += mem.ty.size

            if ty.align < mem.ty.align:
                ty.align = mem.ty.align

            mem = mem.next

        ty.size = self.align_to(offset, ty.align)

        if tag is not None:
            self.push_tag_scope(tag, ty)

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
        mem = ty.members
        while mem is not None:
            if mem.name == name:
                return mem
            mem = mem.next
        return None

    def struct_ref(self, lhs):
        self.add_type(lhs)
        if lhs.ty.kind != TypeKind.TY_STRUCT:
            raise Exception('不是一个结构体')

        tok = self.current_token
        mem = self.find_member(lhs.ty, tok.value)
        self.eat(TokenType.ID)
        if mem is None:
            raise Exception('没有这个成员')

        node = self.new_unary(NodeKind.ND_MEMBER, lhs, tok)
        node.member = mem
        return node

    def gen_addr(self, node):
        if node.kind == NodeKind.ND_VAR:
            if node.var.is_local:
                print("  lea rax, [rbp-%d]" % node.var.offset)
                print("  push rax")
            else:
                print("  push offset %s" % node.var.name)
            return
        if node.kind == NodeKind.ND_DEREF:
            self.code_gen(node.lhs)
            return
        if node.kind == NodeKind.ND_MEMBER:
            self.gen_addr(node.lhs)
            print("  pop rax")
            print("  add rax, %d" % node.member.offset)
            print("  push rax")
            return
        
        self.error(
            error_code=ErrorCode.UNEXPECTED_TOKEN,
            token=self.current_token,
        )

    def gen_lvar(self, node):
        if node.ty.kind == TypeKind.TY_ARRAY:
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
        vl = prog.globals
        while vl is not None:
            var = vl.var
            print("%s:" % var.name)
            if not var.contents:
                print("  .zero %d" % var.ty.size)
                vl = vl.next
                continue
            for c in var.contents:
                print("  .byte %d" % ord(c))
            vl = vl.next

    def align_to(self, n, align):
        return (n + align - 1) & ~(align - 1)

    def code_gen(self, node):
        if node.kind == NodeKind.ND_NULL:
            return
        if node.kind == NodeKind.ND_NUM:
            print("  push %s" % node.val)
            return
        if node.kind == NodeKind.ND_EXPR_STMT:
            self.code_gen(node.lhs)
            print("  add rsp, 8")
            return
        if node.kind == NodeKind.ND_VAR or node.kind == NodeKind.ND_MEMBER:
            self.gen_addr(node)
            if node.ty.kind != TypeKind.TY_ARRAY:
                self.load(node.ty)
            return
        if node.kind == NodeKind.ND_ASSIGN:
            self.gen_lvar(node.lhs)
            self.code_gen(node.rhs)
            self.store(node.ty)
            return
        if node.kind == NodeKind.ND_ADDR:
            self.gen_addr(node.lhs)
            return
        if node.kind == NodeKind.ND_DEREF:
            self.code_gen(node.lhs)
            if node.ty.kind != TypeKind.TY_ARRAY:
                self.load(node.ty)
            return
        if node.kind == NodeKind.ND_IF:
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
        if node.kind == NodeKind.ND_WHILE:
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
        if node.kind == NodeKind.ND_FOR:
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
        if node.kind == NodeKind.ND_BLOCK or node.kind == NodeKind.ND_STMT_EXPR:
            n = node.body
            while n is not None:
                self.code_gen(n)
                n = n.next
            return
        if node.kind == NodeKind.ND_FUNCALL:
            nargs = 0
            arg = node.args
            while arg is not None:
                self.code_gen(arg)
                nargs = nargs + 1
                arg = arg.next
            for i in range(nargs-1, -1, -1):
                print("  pop %s" % argreg8[i])
            # 函数调用前，必须将 RSP 对齐到 16 字节的边界
            # 这是 ABI 的要求
            seq = self.labelseq
            self.labelseq += 1
            print("  mov rax, rsp")
            print("  and rax, 15") # 将rax寄存器填满，也就是对齐操作
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
        if node.kind == NodeKind.ND_RETURN:
            self.code_gen(node.lhs)
            print("  pop rax")
            print("  jmp .L.return.%s" % self.funcname)
            return

        self.code_gen(node.lhs)
        self.code_gen(node.rhs)

        print("  pop rdi")
        print("  pop rax")

        if node.kind == NodeKind.ND_ADD:
            print("  add rax, rdi")
        if node.kind == NodeKind.ND_PTR_ADD:
            print("  imul rdi, %d" % node.ty.base.size)
            print("  add rax, rdi")
        if node.kind == NodeKind.ND_SUB:
            print("  sub rax, rdi")
        if node.kind == NodeKind.ND_PTR_SUB:
            print("  imul rdi, %d" % node.ty.base.size)
            print("  sub rax, rdi")
        if node.kind == NodeKind.ND_PTR_DIFF:
            print("  sub rax, rdi")
            print("  cqo")
            print("  mov rdi, %d" % node.lhs.ty.base.size)
            print("  idiv rdi")
        if node.kind == NodeKind.ND_MUL:
            print("  imul rax, rdi")
        if node.kind == NodeKind.ND_DIV:
            print("  cqo")
            print("  idiv rdi")
        if node.kind == NodeKind.ND_EQ:
            print("  cmp rax, rdi")
            print("  sete al")
            print("  movzb rax, al")
        if node.kind == NodeKind.ND_NE:
            print("  cmp rax, rdi")
            print("  setne al")
            print("  movzb rax, al")
        if node.kind == NodeKind.ND_LT:
            print("  cmp rax, rdi")
            print("  setl al")
            print("  movzb rax, al")
        if node.kind == NodeKind.ND_LE:
            print("  cmp rax, rdi")
            print("  setle al")
            print("  movzb rax, al")

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
        fn = prog.fns
        while fn is not None:
            offset = 0
            # 局部变量是顺序进入数组的，所以需要逆序弹出算偏移量。
            vl = fn.locals
            while vl is not None:
                var = vl.var
                offset = parser.align_to(offset, var.ty.align)
                offset += var.ty.size
                var.offset = offset
                vl = vl.next
            fn.stack_size = parser.align_to(offset, 8)
            fn = fn.next
        print(".intel_syntax noprefix")
        parser.emit_data(prog)
        print(".text")
        fn = prog.fns
        while fn is not None:
            print(".global %s" % fn.name)
            print("%s:" % fn.name)
            parser.funcname = fn.name

            print("  push rbp")
            print("  mov rbp, rsp")
            print("  sub rsp, %d" % fn.stack_size)

            i = 0
            vl = fn.params
            while vl is not None:
                parser.load_arg(vl.var, i)
                i = i + 1
                vl = vl.next

            node = fn.node
            while node is not None:
                parser.code_gen(node)
                node = node.next
            print(".L.return.%s:" % parser.funcname)
            print("  mov rsp, rbp")
            print("  pop rbp")
            print("  ret")
            fn = fn.next
        sys.exit(0)
    except (LexerError, ParserError) as e:
        print(e.message)
        sys.exit(1)

if __name__ == '__main__':
    main()